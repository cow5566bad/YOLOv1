# Download dataset from Dropbox
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget
--quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate
'https://docs.google.com/uc?export=download&id=1dWX3wxwH4F9WRRk2GZJLHNRW5mv4HnPk' -O- | sed -rn
's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dWX3wxwH4F9WRRk2GZJLHNRW5mv4HnPk" -O train_val.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./train_val.zip

# Remove the downloaded zip file
rm ./train_val.zip
