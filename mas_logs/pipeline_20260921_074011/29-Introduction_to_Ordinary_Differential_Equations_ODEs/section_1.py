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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Core Intuition: Rates of Change", [
            "Derivatives describe a rate of change.",
            "Rates often depend on the quantity itself.",
            "Example: Population growth depends on population size."
        ])
        
        # Load asset
        population_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Derivatives describe a rate of change.
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        slope_line = Line(start=self.grid["F6"], end=self.grid["A1"], color="#FF0000")
        self.play(Create(slope_line))
        
        # Place icon at start of slope
        self.place_at_grid(population_icon, "F6", scale_factor=0.3)
        self.play(FadeIn(population_icon))
        
        # === Animation for Lecture Line 2 ===
        # Rates often depend on the quantity itself.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        formula = MathTex(r"\\frac{dy}{dt} = ky", color="#FFFF00")
        self.place_at_grid(formula, "C2", scale_factor=1.0)
        self.play(Write(formula))
        
        dot = Dot(color="#00FF00")
        dot.move_to(slope_line.point_from_proportion(0))
        self.add(dot)
        self.play(MoveAlongPath(dot, slope_line), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        # Example: Population growth depends on population size.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        highlight_box = Rectangle(color="#00FFFF", width=4, height=1)
        self.place_in_area(highlight_box, 'C2', 'C4', scale_factor=0.8)
        self.play(Create(highlight_box))
        
        # Grouping assets for balance
        visual_group = VGroup(population_icon, highlight_box)
        self.place_in_area(visual_group, 'D3', 'F5', scale_factor=0.9)
        
        self.wait(2)
