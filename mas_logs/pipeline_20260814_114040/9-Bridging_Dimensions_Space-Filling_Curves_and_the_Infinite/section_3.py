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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Infinite vs. Finite Boundary", [
            "Finite steps remain one-dimensional.",
            "At infinity, it fills area.",
            "The limit defies finite intuition."
        ])
        
        # Load asset
        square_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        
        # Use place_in_area as recommended by critic (B4, E6 for balance and clutter)
        grid = square_asset.copy().set_stroke(color=BLUE, width=2).set_fill(opacity=0)
        self.place_in_area(grid, 'B4', 'E6', scale_factor=0.7)
        self.add(grid)
        
        # Finite path (approximate)
        path = VMobject(color=YELLOW)
        path.set_points_smoothly([grid.get_corner(DL), grid.get_center(), grid.get_corner(UR)])
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(path), run_time=1.5)
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Represent filling the area
        fill = square_asset.copy().set_color(GREEN).set_fill(opacity=0.5, color=GREEN).set_stroke(width=0)
        self.place_in_area(fill, 'B4', 'E6', scale_factor=0.65)
        self.play(FadeIn(fill), run_time=1.5)
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(path), FadeOut(grid), FadeOut(fill), run_time=1.5)
        self.lecture[2].set_color(RED)
