import gzip

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

def load_test_data():
    for l in readGz("test_Category.json.gz"):
        print l.keys()
        break

def load_data():
    business_ids = {}
    user_ids = {}
    user_business = {}
    user_b_rating = {}
    count = 0
    for l in readGz("train.json.gz"):
        # print l.keys()
        count += 1
        business_ids[l['businessID']] = True
        user_ids[l['userID']] = True
        if l['userID'] in user_business:
            user_business[ l['userID'] ].append( l['businessID'] )
            user_b_rating[ l['userID'] ].append( float(l['rating']) )
        else:
            user_business[ l['userID'] ]= [ l['businessID'] ]
            user_b_rating[ l['userID'] ]= [ float(l['rating']) ]

        # if count > 1000:
        #     break

    b_idx_to_hash = [b_hash for b_hash in business_ids]
    b_hash_to_idx = {b_hash : i for i, b_hash in enumerate(b_idx_to_hash)}

    u_idx_to_hash = [u_hash for u_hash in user_ids]
    u_hash_to_idx = {u_hash : i for i, u_hash in enumerate(u_idx_to_hash)}

    print "num_bus", len(b_idx_to_hash)
    print "num_us", len(u_idx_to_hash)

    return {
        'b_idx_to_hash' : b_idx_to_hash,
        'b_hash_to_idx' : b_hash_to_idx,
        'u_idx_to_hash' : u_idx_to_hash,
        'u_hash_to_idx' : u_hash_to_idx,
        'user_business' : user_business,
        'num_users' : len(u_idx_to_hash),
        'num_business' : len(b_idx_to_hash),
        'user_b_rating' : user_b_rating
    }

def main():
    load_test_data()

if __name__ == '__main__':
    main()