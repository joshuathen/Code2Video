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
            "We shrink the time interval toward zero.",
            "Secant line connects two points.",
            "As points approach, the interval closes.",
            "The secant line becomes a tangent line.",
            "This reveals the rate at one instant."
        ]
        self.setup_layout("The Limit Approach: From Secant to Tangent", lecture_lines)
        
        # Setup Axes and Curve
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], tips=False).scale(0.5)
        curve = axes.plot(lambda x: 0.25 * x**2, color="#FFFF00")
        curve_group = VGroup(axes, curve)
        # Positioned per issue 24 for better use of frame space
        self.place_in_area(curve_group, 'B3', 'F6', scale_factor=0.8)

        # Mobjects
        point_a = Dot(axes.c2p(1, 0.25), color="#00FFFF")
        point_b = Dot(axes.c2p(3, 2.25), color="#00FFFF")
        
        # Persistent Line using ValueTracker instead of always_redraw to stay within budget
        t_a = 1
        t_b_val = 3
        t_b = ValueTracker(t_b_val)
        
        def get_secant():
            return Line(axes.c2p(t_a, 0.25*t_a**2), 
                        axes.c2p(t_b.get_value(), 0.25*t_b.get_value()**2), 
                        color="#FF00FF")
        
        secant = get_secant()
        
        point_b.add_updater(lambda m: m.move_to(axes.c2p(t_b.get_value(), 0.25 * t_b.get_value()**2)))
        secant.add_updater(lambda m: m.become(get_secant()))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(curve_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.add(point_a, point_b, secant)
        self.play(FadeIn(point_a), FadeIn(point_b), Create(secant))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(t_b.animate.set_value(1.1), run_time=3)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(t_b.animate.set_value(1.01), run_time=2)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.wait(2)
