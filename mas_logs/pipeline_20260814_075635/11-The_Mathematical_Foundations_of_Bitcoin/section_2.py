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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Cryptographic Hashing: The Digital Fingerprint", [
            "Hashing is a one-way mathematical function.",
            "Input any data to get a unique fingerprint.",
            "Changing even one bit changes the output entirely."
        ])
        
        # Load asset
        fingerprint_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fingerprint.svg")
        
        # Elements
        input_data = Text("Input: 'Apple'", color=WHITE)
        self.place_at_grid(input_data, "B2", scale_factor=0.6)
        
        blender = Rectangle(color=GRAY, height=1.5, width=1.5)
        blender_text = Text("SHA-256", color=WHITE).scale(0.5).move_to(blender.get_center())
        blender_group = VGroup(blender, blender_text)
        self.place_at_grid(blender_group, "C3", scale_factor=0.7)
        
        output_hash = Text("Hash: 'a5c9...'", color=WHITE)
        self.place_at_grid(output_hash, "D4", scale_factor=0.6)
        
        # Asset representation
        input_asset = fingerprint_asset.copy().set_color(WHITE)
        self.place_at_grid(input_asset, "B4", scale_factor=0.3)
        
        output_asset = fingerprint_asset.copy().set_color("#FF00FF")
        self.place_at_grid(output_asset, "D4", scale_factor=0.3)
        output_asset.next_to(output_hash, RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(input_data), Write(input_asset), Write(blender_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.play(
            ReplacementTransform(input_asset.copy(), output_asset),
            output_hash.animate.set_color("#FF00FF")
        )
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        glow = SurroundingRectangle(output_asset, color="#00FFFF", buff=0.1)
        self.play(Create(glow))
        self.wait(2)
