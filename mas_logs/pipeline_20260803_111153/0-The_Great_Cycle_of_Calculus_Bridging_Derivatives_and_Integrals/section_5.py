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
            "The Fundamental Theorem of Calculus bridges slope and area.",
            "It connects the derivative of a function to its integral.",
            "The definite integral calculates net change over an interval.",
            "Evaluation uses the anti-derivative at the boundary points.",
            "This theorem unites the two main branches of calculus."
        ]
        self.setup_layout("The Fundamental Theorem: The Bridge", lecture_lines)
        
        # Dim all lecture lines initially
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # The Fundamental Theorem of Calculus bridges slope and area.
        # FTC formula \int_a^b f(x)dx = F(b) - F(a) appears in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(WHITE))
        formula = MathTex(r"\int_a^b f(x)dx = F(b) - F(a)", color=WHITE)
        # Resolved issue 43: Moving formula to E4-F6
        self.place_in_area(formula, 'E4', 'F6', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It connects the derivative of a function to its integral.
        # The screen splits into two panels: Panel A (left) and Panel B (right).
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Panel A: Rate Function f(x) = 2x
        ax_a = Axes(x_range=[0, 3], y_range=[0, 6], x_length=2.5, y_length=2.5, 
                   axis_config={"include_tip": False}).set_color(GRAY)
        label_a = Text("Rate: f(x)", font_size=16, color=WHITE)
        panel_a = VGroup(ax_a, label_a)
        label_a.next_to(ax_a, UP, buff=0.1)
        # Resolved issue 41: Moving panel_a to A3-C4
        self.place_in_area(panel_a, 'A3', 'C4', scale_factor=0.7)
        
        # Panel B: Accumulation Function F(x) = x^2
        ax_b = Axes(x_range=[0, 3], y_range=[0, 9], x_length=2.5, y_length=2.5,
                   axis_config={"include_tip": False}).set_color(GRAY)
        label_b = Text("Accumulation: F(x)", font_size=16, color=WHITE)
        panel_b = VGroup(ax_b, label_b)
        label_b.next_to(ax_b, UP, buff=0.1)
        # Resolved issue 42: Moving panel_b to A5-C6
        self.place_in_area(panel_b, 'A5', 'C6', scale_factor=0.7)
        
        self.play(Create(ax_a), Create(label_a), Create(ax_b), Create(label_b))
        
        f_curve = ax_a.plot(lambda x: 2*x, x_range=[0, 2.5], color=WHITE)
        F_curve = ax_b.plot(lambda x: x**2, x_range=[0, 2.5], color=WHITE)
        self.play(Create(f_curve), Create(F_curve))
        self.wait(1)

        # Prepare for synchronization using ValueTracker
        tracker = ValueTracker(0.01)
        
        # Area under f(x) - Persistent mobject using updater
        # This represents the integral accumulation
        area = VMobject(color="#00FF00", fill_opacity=0.5, stroke_width=0)
        area.add_updater(lambda m: m.set_points_as_corners([
            ax_a.c2p(0, 0),
            *[ax_a.c2p(x, 2*x) for x in np.linspace(0, tracker.get_value(), 20)],
            ax_a.c2p(tracker.get_value(), 0),
            ax_a.c2p(0, 0)
        ]))

        # Point on F(x) - Persistent mobject using updater
        # This represents the value of the anti-derivative
        dot_b = Dot(color="#00BFFF")
        dot_b.add_updater(lambda d: d.move_to(ax_b.c2p(tracker.get_value(), tracker.get_value()**2)))

        # === Animation for Lecture Line 3 ===
        # The definite integral calculates net change over an interval.
        # Panel A shows area under f(x) filling up in #00FF00.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.add(area)
        self.play(tracker.animate.set_value(1.5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Evaluation uses the anti-derivative at the boundary points.
        # Panel B shows a point on the graph of F(x) rising in #00BFFF.
        self.play(self.lecture[3].animate.set_color("#00BFFF"))
        self.add(dot_b)
        self.play(Indicate(dot_b), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This theorem unites the two main branches of calculus.
        # A bridge icon in #FFD700 connects the panels as they animate in sync.
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        # Representing the FTC "Bridge" visually
        bridge = DoubleArrow(
            start=ax_a.get_right(), 
            end=ax_b.get_left(), 
            color="#FFD700", 
            buff=0.1, 
            stroke_width=4
        )
        bridge_label = Text("FTC BRIDGE", font_size=14, color="#FFD700").next_to(bridge, UP, buff=0.05)
        
        self.play(Create(bridge), Write(bridge_label))
        
        # Final synchronous animation showing the link between area and value
        self.play(tracker.animate.set_value(2.2), run_time=2, rate_func=linear)
        self.wait(2)
