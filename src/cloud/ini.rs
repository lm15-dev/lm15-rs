//! The INI reader the AWS profile files need (`configparser` semantics as
//! botocore reads them: `[section]` headers, `key = value` with `=` or
//! `:`, `#` / `;` comment lines, keys lowercased, values trimmed, a
//! later duplicate wins). No interpolation.

use std::collections::BTreeMap;

use crate::auth::AuthError;

pub type Section = BTreeMap<String, String>;

#[derive(Debug, Default, Clone)]
pub struct Ini {
    sections: BTreeMap<String, Section>,
}

impl Ini {
    pub fn parse(text: &str) -> Result<Ini, AuthError> {
        let mut ini = Ini::default();
        let mut current: Option<String> = None;
        for raw in text.lines() {
            let line = raw.trim_end();
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
                continue;
            }
            if trimmed.starts_with('[') {
                let Some(end) = trimmed.find(']') else {
                    return Err(malformed());
                };
                let name = trimmed[1..end].trim().to_string();
                ini.sections.entry(name.clone()).or_default();
                current = Some(name);
                continue;
            }
            // A continuation line (indented, no separator) extends the
            // previous value in configparser; botocore's nested tables
            // (`s3 =\n  ...`) are read the same way and are not needed here.
            let Some(section) = &current else {
                return Err(malformed());
            };
            let Some(split) = trimmed.find(['=', ':']) else {
                if line.starts_with(char::is_whitespace) {
                    continue;
                }
                return Err(malformed());
            };
            let key = trimmed[..split].trim().to_ascii_lowercase();
            let value = trimmed[split + 1..].trim().to_string();
            if key.is_empty() {
                return Err(malformed());
            }
            ini.sections
                .get_mut(section)
                .expect("the current section exists")
                .insert(key, value);
        }
        Ok(ini)
    }

    pub fn has_section(&self, name: &str) -> bool {
        self.sections.contains_key(name)
    }

    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.get(name)
    }
}

fn malformed() -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message: "malformed AWS profile configuration; check the AWS config and credentials files"
            .into(),
        hint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_profiles_and_sso_sessions() {
        let ini = Ini::parse(
            "# comment\n[default]\nregion = us-east-1\n\n[profile work]\nsso_session = corp\n\
             SSO_ACCOUNT_ID: 1111\n[sso-session corp]\nsso_start_url = https://x\n",
        )
        .unwrap();
        assert_eq!(ini.section("default").unwrap()["region"], "us-east-1");
        assert_eq!(ini.section("profile work").unwrap()["sso_session"], "corp");
        assert_eq!(
            ini.section("profile work").unwrap()["sso_account_id"],
            "1111"
        );
        assert!(ini.has_section("sso-session corp"));
        assert!(Ini::parse("key = value\n").is_err());
        assert!(Ini::parse("[broken\n").is_err());
    }
}
