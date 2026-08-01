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
        # Setup data
        title = "The Derivative: The Infinite Zoom"
        lines = [
            "Curves have slopes that change at every single point.",
            "To find the exact slope, we zoom in infinitely.",
            "At extreme magnification, a curve looks like a line.",
            "This straight line is called the tangent line.",
            "It reveals the object's speed at that exact moment."
        ]
        self.setup_layout(title, lines)
        
        # Color definitions
        COLOR_TRACK = "#FF00FF"    # Magenta
        COLOR_MAGNIFIER = "#FFFFFF" # White
        COLOR_TANGENT = "#FFFF00"   # Yellow
        COLOR_TEXT = "#FFFFFF"      # White
        
        # === Animation for Lecture Line 1 ===
        # Display a curved roller coaster track (#FF00FF) featuring a roller [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/roller.svg].
        self.lecture[0].set_color(COLOR_TRACK)
        
        axes = Axes(
            x_range=[-2, 2], 
            y_range=[-1, 3], 
            axis_config={"include_tip": False, "stroke_width": 2, "color": GREY_C}
        ).scale(0.6)
        
        # f(x) = 0.5x^2 + 1 represents the curved track
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, x_range=[-2, 2], color=COLOR_TRACK)
        
        # Asset: Load and place roller SVG
        roller = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/roller.svg")
        roller.set_color(COLOR_TRACK).scale(0.2)
        # Place roller on the curve at x = -1.2
        roller_pos = axes.input_to_graph_point(-1.2, curve)
        roller.move_to(roller_pos)
        
        track_group = VGroup(axes, curve, roller)
        self.place_in_area(track_group, "B2", "E5")
        
        self.play(Create(axes), Create(curve), FadeIn(roller), run_time=1.5)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # To find the exact slope, we zoom in infinitely.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_MAGNIFIER)
        
        # Point to zoom in on (x=1)
        zoom_point_coords = axes.input_to_graph_point(1.0, curve)
        
        magnifier_circle = Circle(radius=0.4, color=COLOR_MAGNIFIER, stroke_width=4)
        magnifier_handle = Line(ORIGIN, 0.3*DOWN + 0.3*RIGHT, color=COLOR_MAGNIFIER, stroke_width=4)
        magnifier_handle.next_to(magnifier_circle, DR, buff=-0.1)
        magnifier = VGroup(magnifier_circle, magnifier_handle)
        magnifier.move_to(zoom_point_coords)
        
        self.play(FadeIn(magnifier, scale=1.5), run_time=1)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # At extreme magnification, a curve looks like a line.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_TEXT)
        
        # Create a highly zoomed-in version of the same curve
        # At x=1, y=1.5. Range [0.9, 1.1] is 0.2 units wide.
        zoomed_axes = Axes(
            x_range=[0.9, 1.1], 
            y_range=[1.4, 1.6],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": False, "stroke_width": 1, "color": GREY_E}
        ).scale(0.5)
        
        zoomed_curve = zoomed_axes.plot(lambda x: 0.5 * x**2 + 1, x_range=[0.9, 1.1], color=COLOR_TRACK)
        
        # Place the roller on the zoomed curve at x=1.0
        zoomed_roller = roller.copy().scale(1.5)
        zoomed_roller_pos = zoomed_axes.input_to_graph_point(1.0, zoomed_curve)
        zoomed_roller.move_to(zoomed_roller_pos)
        
        zoomed_view = VGroup(zoomed_axes, zoomed_curve, zoomed_roller)
        self.place_in_area(zoomed_view, "B2", "E5")
        
        # Visual zoom effect: scale up the original track group and transform it
        self.play(
            FadeOut(magnifier),
            ReplacementTransform(track_group, zoomed_view),
            run_time=2
        )
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # This straight line is called the tangent line.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_TANGENT)
        
        # The tangent line at x=1 for f(x)=0.5x^2 + 1 is y = x + 0.5
        tangent_line = zoomed_axes.plot(lambda x: x + 0.5, x_range=[0.9, 1.1], color=COLOR_TANGENT)
        tangent_label = Text("Tangent Line", font_size=24, color=COLOR_TANGENT)
        # VideoCritic fix: place_in_area 'A5' to 'B6' scale 0.7
        self.place_in_area(tangent_label, "A5", "B6", scale_factor=0.7)
        
        self.play(Create(tangent_line), Write(tangent_label), run_time=1.5)
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # It reveals the object's speed at that exact moment.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_TEXT)
        
        diff_text = Text("Differentiation", font_size=36, color=COLOR_TEXT)
        # VideoCritic fix: place_in_area 'F1' to 'F6' scale 0.7 to avoid cut-off and clutter
        self.place_in_area(diff_text, "F1", "F6", scale_factor=0.7)
        
        self.play(Write(diff_text), run_time=1.5)
        self.wait(2)
