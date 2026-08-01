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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the title and lecture lines
        lecture_lines = [
            "Consider the Folium of Descartes at point (3, 3).",
            "Apply the recipe to find the slope here.",
            "The robotic hand moves with a slope of negative one."
        ]
        self.setup_layout("Guided Application: The Robotic Arm", lecture_lines)

        # Colors for visual consistency
        color_step1 = "#009E73"  # Bluish Green
        color_step2 = "#56B4E9"  # Sky Blue
        color_step3 = "#D55E00"  # Vermillion (Red)

        # === Animation for Lecture Line 1 ===
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "font_size": 18}
        )
        
        # Implicit curve for x^3 + y^3 - 6xy = 0
        folium_curve = ImplicitFunction(
            lambda x, y: x**3 + y**3 - 6*x*y,
            x_range=[-1, 5],
            y_range=[-1, 5],
            color=color_step1
        )
        
        # Point (3,3) and label
        point_33 = Dot(axes.c2p(3, 3), color=color_step3)
        label_33 = Text("(3, 3)", font_size=20, color=color_step3)
        # Manually placing label near point relative to axes
        label_33.next_to(point_33, UR, buff=0.1)

        # Tangent line: y = -x + 6 (slope -1 through (3,3))
        tangent_line = Line(
            start=axes.c2p(1.5, 4.5),
            end=axes.c2p(4.5, 1.5),
            color=color_step3,
            stroke_width=4
        )

        # Group all graph elements to position them correctly in the grid
        graph_group = VGroup(axes, folium_curve, point_33, label_33, tangent_line)
        self.place_in_area(graph_group, "A2", "D5")

        self.play(self.lecture[0].animate.set_color(color_step1))
        self.play(Create(axes), Create(folium_curve))
        self.play(FadeIn(point_33), Write(label_33))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_step2))
        
        # Derived formula dy/dx
        formula = MathTex(
            r"\frac{dy}{dx} = \frac{2y - x^2}{y^2 - 2x}", 
            font_size=28, 
            color=color_step2
        )
        self.place_in_area(formula, "E2", "E5")
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_step3))
        
        # Substitute (3,3) into formula
        calc = MathTex(
            r"\text{slope} = \frac{2(3) - 3^2}{3^2 - 2(3)} = -1",
            font_size=24,
            color=color_step3
        )
        self.place_in_area(calc, "F2", "F5")
        
        self.play(Write(calc))
        self.play(Create(tangent_line))
        self.wait(2)
