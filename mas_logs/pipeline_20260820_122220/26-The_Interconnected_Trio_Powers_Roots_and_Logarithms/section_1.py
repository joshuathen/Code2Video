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
        lecture_lines = [
            "Powers represent exponential growth over time.",
            "Consider population doubling: base 2, time x.",
            "After 3 months, the population is 8.",
            "The formula 2^x = y defines growth.",
            "Base b raised to power x is y."
        ]
        self.setup_layout("The Foundation: Understanding Exponential Growth", lecture_lines)
        
        # Mobjects
        dot = Dot(color=WHITE)
        initial_label = Text("Initial", font_size=18, color=WHITE)
        self.place_at_grid(dot, 'F1', scale_factor=1.0)
        self.place_at_grid(initial_label, 'F2', scale_factor=0.8)
        initial_label.next_to(dot, RIGHT)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(dot), Write(initial_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        # Using area for the curve group as requested by Critic
        growth_curve = FunctionGraph(lambda x: 2**(x+1) * 0.2, x_range=[0, 3], color="#FF4500")
        curve_label = Text("y = 2^x", font_size=18, color="#FF4500")
        curve_group = VGroup(growth_curve, curve_label)
        self.place_in_area(curve_group, 'C3', 'F6', scale_factor=0.8)
        self.play(Create(growth_curve), Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Load asset
        population_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        # Fix: Grid C5 requested by Critic
        self.place_at_grid(population_icon, 'C5', scale_factor=0.5)
        # Fix: Formula placement requested by Critic
        # Actually Line 3 needs point at 3,8. Let's place it at D4
        target_point = Dot(color="#FFFF00").move_to(self.grid['D4'])
        month_label = Text("3 Months", font_size=18, color="#FFFF00")
        month_label.next_to(target_point, UP)
        
        self.play(FadeIn(target_point), FadeIn(population_icon), Write(month_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Formula placement fix requested by Critic: B1-B6
        formula = MathTex("2^x = y", color=WHITE)
        self.place_in_area(formula, 'B1', 'B6', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        # Placeholder or summary
        final_label = Text("Growth Defined", font_size=18, color=WHITE)
        self.place_at_grid(final_label, 'A3', scale_factor=0.8)
        self.play(FadeIn(final_label))
        self.wait(2)
