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
        lecture_lines = [
            "We shrink the interval between two points.",
            "The secant line approaches the tangent.",
            "This transformation defines the derivative.",
            "Imagine zooming into a cheetah's path.",
            "We find speed at an instant."
        ]
        self.setup_layout("The Dynamic Transition: Shrinking the Interval", lecture_lines)
        
        # Setup graph/points
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False}).scale(0.4)
        curve = axes.plot(lambda x: 0.2 * x**3, color=BLUE)
        
        # B004: Columns 4-6. B025: B2-E4
        self.place_in_area(axes, 'B2', 'E4', scale_factor=0.8)
        
        # Asset: Cheetah Icon
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah, 'A4', scale_factor=0.3)
        
        x0 = 1.0
        h = ValueTracker(2.0)
        p1 = Dot(axes.c2p(x0, 0.2*x0**3), color=RED).scale(0.5)
        p2 = always_redraw(lambda: Dot(axes.c2p(x0 + h.get_value(), 0.2*(x0 + h.get_value())**3), color=GREEN).scale(0.5))
        
        secant = always_redraw(lambda: Line(p1.get_center(), p2.get_center(), color=YELLOW))
        
        self.add(curve, p1, p2, secant, cheetah)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Animate cheetah movement alongside interval
        self.play(
            h.animate.set_value(0.1),
            cheetah.animate.move_to(self.grid['B4']),
            run_time=3
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        formula = MathTex(r"\\frac{f(x+h)-f(x)}{h}", color="#F1C40F").scale(0.7)
        # B024: Place at F5 to avoid overlap
        self.place_at_grid(formula, 'F5', scale_factor=0.7)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(cheetah.animate.move_to(self.grid['F4']), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        # Transform formula at the limit
        limit_text = Tex("f'(x)", color=RED).move_to(formula.get_center())
        self.play(Transform(formula, limit_text), run_time=1)
        self.wait(2)
