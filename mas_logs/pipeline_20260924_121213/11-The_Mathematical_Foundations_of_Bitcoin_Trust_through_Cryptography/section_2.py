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
        lecture_lines = [
            "SHA-256 acts as a digital fingerprint.",
            "Any input produces a fixed-size unique hash.",
            "Changing one bit completely alters the output."
        ]
        self.setup_layout("Cryptographic Hash Functions", lecture_lines)
        
        # Elements
        input_data = Text("Input Data", font_size=24, color="#FFFFFF")
        hash_box = Rectangle(height=1.5, width=2.0, color="#0000FF").set_fill("#0000FF", opacity=0.3)
        hash_text = Text("SHA-256", font_size=20, color="#FFFFFF").move_to(hash_box.get_center())
        hash_func = VGroup(hash_box, hash_text)
        
        # Use asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fingerprint.svg]
        fingerprint_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fingerprint.svg").set_color("#00FF00")
        output_data = VGroup(Text("Digital Fingerprint", font_size=24, color="#00FF00"), fingerprint_icon).arrange(DOWN)
        
        # Position with suggested corrections
        self.place_at_grid(input_data, 'B4', scale_factor=0.8)
        self.place_at_grid(hash_func, 'C4', scale_factor=0.8)
        self.place_at_grid(output_data, 'D4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(input_data), FadeIn(hash_func), FadeIn(output_data))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(Indicate(hash_func))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        # Visualizing the avalanche effect
        bit_change = Text("0 -> 1", font_size=18, color="#FF0000").next_to(input_data, UP)
        new_output = Text("A1B2... (Different)", font_size=20, color="#FF0000").next_to(output_data, DOWN)
        self.play(FadeIn(bit_change), ReplacementTransform(output_data.copy(), new_output))
        self.wait(2)
