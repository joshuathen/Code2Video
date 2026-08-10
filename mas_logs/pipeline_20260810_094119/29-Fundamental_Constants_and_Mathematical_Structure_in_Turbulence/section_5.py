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
            "Inertial scaling balances viscous dissipation perfectly.",
            "This balance is crucial for CFD accuracy.",
            "Mathematical structures remain consistent across scales."
        ]
        self.setup_layout("Conclusion and Real-World Limits", lecture_lines)
        
        # Load SVG Assets - Disabled cache to prevent FileNotFoundError on generated temp files
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg", use_svg_cache=False)
        turbine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turbine.svg", use_svg_cache=False)
        placeholder = Square(side_length=0.8, color=GREEN)

        # === Animation for Lecture Line 1 ===
        # Review real-world applications of turbulence involving airplane.svg
        self.place_at_grid(airplane, 'B4', scale_factor=0.6)
        self.play(FadeIn(airplane))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show limitation of Kolmogorov theory in boundary layers
        self.place_at_grid(placeholder, 'D4', scale_factor=0.8)
        self.play(Create(placeholder))
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Conclude with summary of modern turbulence research using turbine.svg
        self.place_at_grid(turbine, 'D6', scale_factor=0.6)
        self.play(FadeIn(turbine))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
