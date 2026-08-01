from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Hashing creates a unique fingerprint for data.",
            "Even small input changes drastically alter the hash.",
            "This ensures no two IDs produce the same hash.",
            "Servers use hashes, not direct identifiers.",
            "This is crucial for privacy."
        ]
        self.setup_layout("Mathematical Underpinnings (Visualized)", lecture_lines)
        
        # === Animation for Lecture Line 1: Hashing creates a unique fingerprint for data. ===
        # Visualize a simple input and its resulting hash.
        input_data = Text("Input Data", font_size=24)
        hash_output = Text("Hash Output", font_size=24)
        # Simplified visualization of a hashing process. In a real scenario, this would be more complex.
        hash_process_arrow = Arrow(start=LEFT, end=RIGHT, fill_opacity=0.5)

        self.play(Write(input_data))
        self.place_at_grid(input_data, "B2")
        
        # Create a mock hash output and place it
        mock_hash_output = Text("Unique Hash", font_size=24).scale(0.8)
        self.place_at_grid(mock_hash_output, "B4")
        self.play(FadeIn(mock_hash_output))

        # === Animation for Lecture Line 2: Even small input changes drastically alter the hash. ===
        # Show that changing the input slightly results in a completely different hash.
        changed_input_data = Text("Input Data (Slightly Changed)", font_size=24)
        changed_hash_output = Text("Completely Different Hash", font_size=24).scale(0.8)
        
        changed_input_data.move_to(self.grid["D2"])
        changed_hash_output.move_to(self.grid["D4"])

        self.play(FadeOut(input_data), FadeOut(mock_hash_output)) # Clear previous elements
        self.play(Write(changed_input_data))
        
        # Create a mock hash output for the changed input and place it
        mock_changed_hash_output = Text("Completely Different Hash", font_size=24).scale(0.8)
        mock_changed_hash_output.move_to(self.grid["D4"])
        self.play(FadeIn(mock_changed_hash_output))

        # === Animation for Lecture Line 3: This ensures no two IDs produce the same hash. ===
        # Reiterate the uniqueness aspect by showing two distinct hashes.
        id1_hash = Text("Hash A", font_size=24).scale(0.8)
        id2_hash = Text("Hash B", font_size=24).scale(0.8)
        
        id1_hash.move_to(self.grid["E2"])
        id2_hash.move_to(self.grid["E4"])
        
        self.play(FadeOut(changed_input_data), FadeOut(mock_changed_hash_output)) # Clear previous elements
        self.play(Write(id1_hash))
        self.play(Write(id2_hash))
        # Visually imply they are different and unique (no direct animation here, just placement)

        # === Animation for Lecture Line 4: Servers use hashes, not direct identifiers. ===
        # Show a server icon receiving hashes.
        # Using a placeholder for SVGMobject as actual SVG files are not directly handled here.
        # In a real scenario, you would ensure '/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg' is available or use a Manim primitive.
        server_icon = Square(side_length=0.8, color=WHITE, fill_opacity=0.5) 
        server_icon.move_to(self.grid["F4"])
        
        hash1_for_server = id1_hash.copy().scale(0.5)
        hash2_for_server = id2_hash.copy().scale(0.5)
        
        hash1_for_server.move_to(self.grid["E1"])
        hash2_for_server.move_to(self.grid["F1"])

        self.play(FadeOut(id1_hash), FadeOut(id2_hash)) # Clear previous elements
        self.play(FadeIn(server_icon), Write(hash1_for_server), Write(hash2_for_server))
        self.play(hash1_for_server.animate.move_to(server_icon.get_center() + LEFT * 0.5),
                  hash2_for_server.animate.move_to(server_icon.get_center() + RIGHT * 0.5))
        self.play(FadeOut(hash1_for_server), FadeOut(hash2_for_server))

        # === Animation for Lecture Line 5: This is crucial for privacy. ===
        # Conclude with a privacy icon or statement.
        privacy_icon = Text("🔒", font_size=48) # Simple emoji for privacy
        privacy_icon.move_to(self.grid["C5"])
        
        self.play(FadeOut(server_icon)) # Clear server
        self.play(FadeIn(privacy_icon))
        self.play(FadeOut(privacy_icon))

        self.wait(1)
