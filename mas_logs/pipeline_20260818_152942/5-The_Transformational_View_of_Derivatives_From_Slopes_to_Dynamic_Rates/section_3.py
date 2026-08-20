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
        lecture_lines = [
            "Derivatives provide a local linear approximation.",
            "Zooming in reveals the curve's linear nature.",
            "Complexity simplifies into a local straight line.",
            "This mapping captures slope at one point.",
            "The derivative is essentially local linearization."
        ]
        self.setup_layout("Conceptual Core: The Derivative as a Local Linearization", lecture_lines)
        
        # Assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # Setup Math objects
        curve = FunctionGraph(lambda x: 0.2 * x**2, x_range=[-3, 3], color=WHITE)
        point = Dot(curve.point_from_proportion(0.5), color=YELLOW)
        tangent = Line(start=point.get_center() - np.array([1, 0, 0]), end=point.get_center() + np.array([1, 0, 0]), color=YELLOW)
        formula = MathTex(r"y = f(a) + f'(a)(x-a)", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(curve, 'A2', 'C4', scale_factor=0.6)
        self.place_in_area(point, 'A2', 'D4', scale_factor=0.6)
        self.play(Create(curve), FadeIn(point))
        self.place_in_area(tangent, 'A2', 'D4', scale_factor=0.6)
        self.play(Create(tangent))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        zoom_view = VGroup(curve.copy(), point.copy()).scale(2)
        self.place_in_area(zoom_view, 'E2', 'F5', scale_factor=0.7)
        glass = magnifying_glass.copy().set_color(WHITE)
        self.place_at_grid(glass, 'E3', scale_factor=0.3)
        self.play(FadeIn(zoom_view), FadeIn(glass))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(formula, 'B5', scale_factor=0.7)
        self.play(Write(formula))
        self.lecture[2].set_color("#00BFFF")

        # === Animation for Lecture Line 4 ===
        self.play(Indicate(tangent))
        self.lecture[3].set_color("#FF4500")

        # === Animation for Lecture Line 5 ===
        formula_new = MathTex(r"y = f(a) + ", r"f'(a)", r"(x-a)", font_size=36)
        formula_new.set_color_by_tex(r"f'(a)", "#32CD32")
        self.place_at_grid(formula_new, 'B5', scale_factor=0.7)
        
        glass_final = magnifying_glass.copy().set_color("#32CD32")
        self.place_at_grid(glass_final, 'C5', scale_factor=0.2)
        
        self.play(ReplacementTransform(formula, formula_new), FadeIn(glass_final))
        self.lecture[4].set_color("#32CD32")
        self.wait(2)
