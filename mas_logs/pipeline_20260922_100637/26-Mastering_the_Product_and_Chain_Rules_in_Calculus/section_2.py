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
        self.setup_layout("The Product Rule: The Collaborative Effort", [
            "Product rule handles multiplying two functions.",
            "Think of a rectangle expanding in area.",
            "Change equals the sum of two strips."
        ])
        
        # Assets
        rect_asset1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rectangle.svg")
        rect_asset2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rectangle.svg")
        
        # Animations
        # Define base rectangle
        rect = rect_asset1
        self.place_at_grid(rect, 'B4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(rect), run_time=1)
        self.lecture[0].set_color("#2ECC71")
        
        # === Animation for Lecture Line 2 ===
        strip1 = Rectangle(width=0.5, height=1, color="#F1C40F", fill_opacity=0.7)
        self.place_at_grid(strip1, 'C4', scale_factor=0.5)
        
        self.play(FadeIn(strip1), run_time=1)
        self.lecture[1].set_color("#F1C40F")
        
        # === Animation for Lecture Line 3 ===
        strip2 = rect_asset2
        strip2.set_color("#E67E22")
        self.place_at_grid(strip2, 'D4', scale_factor=0.5)
        
        self.play(FadeIn(strip2), run_time=1)
        self.lecture[2].set_color("#E67E22")
        self.wait(2)
