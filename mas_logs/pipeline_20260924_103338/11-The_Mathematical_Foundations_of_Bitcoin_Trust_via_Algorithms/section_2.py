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
        self.setup_layout("The Mechanism: Proof of Work", [
            "Mining is a competitive, computational race.",
            "Miners search for hashes below a target.",
            "The nonce acts as the variable to tweak."
        ])
        
        # Assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")

        # === Animation for Lecture Line 1 ===
        # Use computer icon and nonce setup
        nonce_label = Text("Nonce:", font_size=24, color="#FFFFFF")
        nonce_val = DecimalNumber(0, num_decimal_places=0, font_size=30, color="#FFFF00")
        nonce_group = VGroup(nonce_label, nonce_val).arrange(RIGHT)
        
        self.place_at_grid(computer_icon, 'B2', scale_factor=0.3)
        self.place_at_grid(nonce_group, 'C2', scale_factor=0.8)
        
        self.play(FadeIn(computer_icon), Write(nonce_group), FadeToColor(self.lecture[0], "#00FF00"))
        
        # === Animation for Lecture Line 2 ===
        # Hash search logic
        hash_box = Rectangle(width=2, height=1, color="#FF00FF", fill_opacity=0.3)
        hash_text = Text("Hash", font_size=20, color="#FF00FF")
        hash_obj = VGroup(hash_box, hash_text)
        
        # Fix: Using updated area based on feedback
        self.place_in_area(hash_obj, 'B3', 'C4', scale_factor=0.9)
        self.place_at_grid(server_icon, 'D3', scale_factor=0.3)
        
        target_line = Line(start=LEFT*1, end=RIGHT*1, color="#00FFFF")
        target_label = Text("Target", font_size=20, color="#00FFFF")
        target_group = VGroup(target_line, target_label).arrange(DOWN)
        
        # Fix: Using updated area based on feedback
        self.place_in_area(target_group, 'B5', 'C6', scale_factor=0.9)
        
        self.play(FadeIn(hash_obj), FadeIn(server_icon), FadeIn(target_group), FadeToColor(self.lecture[1], "#00FFFF"))
        
        # === Animation for Lecture Line 3 ===
        target_icon = Star(n=5, color="#00FF00", fill_opacity=1).scale(0.3)
        
        # Fix: Using updated grid based on feedback
        self.place_at_grid(target_icon, 'D5', scale_factor=0.7)
        
        self.play(
            FadeToColor(self.lecture[2], "#FFFF00"),
            nonce_val.animate.set_value(9999),
            FadeIn(target_icon),
            run_time=2
        )
        self.wait(1)
