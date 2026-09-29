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
        self.setup_layout("Data Collection & Pattern Hunting", [
            "Three dots make four regions.", 
            "Four dots create eight regions.", 
            "Five dots give sixteen regions."
        ])
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dots.svg]
        dots = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dots.svg")
        
        # Create Table
        table = VGroup(
            Text("n points", font_size=24), Text("Regions", font_size=24),
            Text("0", font_size=20), Text("1", font_size=20),
            Text("1", font_size=20), Text("2", font_size=20),
            Text("2", font_size=20), Text("4", font_size=20),
            Text("3", font_size=20), Text("8", font_size=20),
            Text("4", font_size=20), Text("16", font_size=20)
        ).arrange_in_grid(rows=6, cols=2, buff=0.5)
        
        # VideoCritic requested fix for grid positioning
        self.place_in_area(table, 'C2', 'F6', scale_factor=0.45)
        
        # === Animation for Lecture Line 1 ===
        # Use dot asset
        self.place_at_grid(dots.copy(), 'A3', scale_factor=0.3)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Use dot asset
        self.place_at_grid(dots.copy(), 'A4', scale_factor=0.3)
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)
