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
        self.setup_layout("Summary and Synthesis", [
            "Derivatives break curves down into slopes.", 
            "Integrals build curves up into areas.", 
            "They are two sides of the same coin."
        ])
        
        # Assets
        slope_line = Line(start=LEFT*0.5, end=RIGHT*0.5, color=YELLOW)
        area = Polygon(ORIGIN, RIGHT*1.5, RIGHT*1.5+UP*0.5, UP*0.5, color=RED).set_fill(RED, opacity=0.5)
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=GOLD)
        
        # === Animation for Lecture Line 1 ===
        # Derivatives break curves down into slopes.
        self.place_at_grid(slope_line, 'D5', scale_factor=0.8)
        self.play(FadeIn(slope_line))
        self.lecture[0].set_color(YELLOW)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Integrals build curves up into areas.
        self.place_at_grid(area, 'D5', scale_factor=0.6)
        self.play(FadeIn(area))
        self.lecture[1].set_color(RED)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # They are two sides of the same coin.
        self.place_at_grid(coin, 'E3', scale_factor=0.9)
        self.play(DrawBorderThenFill(coin))
        self.lecture[2].set_color(GOLD)
        self.wait(2)
