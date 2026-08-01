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
        # Section Title and Lecture Lines (Synchronized 1:1 with 3 steps)
        title = "The Final Destination: pi i"
        lecture_lines = [
            "Now, we travel a distance of exactly pi.",
            "This path covers half of the unit circle's circumference.",
            "We land precisely at negative one on the complex plane."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors for mapping lecture lines to visual elements
        color_travel = "#FFFF00"  # Yellow
        color_half_turn = "#00FFFF" # Cyan
        color_land = "#FF0000"   # Red
        
        # Pre-calculate axes and circle properties to avoid always_redraw
        # Using specific area and scale as per Issue 37
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GRAY}
        )
        self.place_in_area(axes, "B2", "D6", scale_factor=0.8)
        
        # Unit circle radius in plot coordinates
        radius_val = axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]
        circle = Circle(radius=radius_val, color=WHITE, stroke_opacity=0.3)
        circle.move_to(axes.c2p(0, 0))
        
        # Labels for 1 and -1
        label_1 = Text("1", font_size=20, color=WHITE)
        label_1.next_to(axes.c2p(1, 0), UR, buff=0.1)
        
        label_neg1 = Text("-1", font_size=20, color=WHITE)
        label_neg1.next_to(axes.c2p(-1, 0), UL, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        # Description: Show the unit circle with a marker starting at 1 on the real axis.
        self.lecture[0].set_color(color_travel)
        
        marker = Dot(axes.c2p(1, 0), color=color_travel)
        
        self.play(
            Create(axes),
            Create(circle),
            Write(label_1),
            FadeIn(marker)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Description: Move the marker along the top half of the circle's circumference.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_half_turn)
        
        arc_path = Arc(
            radius=radius_val,
            start_angle=0,
            angle=PI,
            color=color_half_turn,
            stroke_width=4
        )
        # Position the arc correctly
        arc_path.move_to(axes.c2p(0, 0), aligned_edge=DOWN)
        
        self.play(
            MoveAlongPath(marker, arc_path),
            Create(arc_path),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Description: The marker lands at -1 on the real axis, highlighting the value in red (#FF0000).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_land)
        
        # Equation from Issue 36: scale 0.9, area E3-F5
        equation = Text("e^iπ = -1", color=color_land, font_size=32)
        self.place_in_area(equation, "E3", "F5", scale_factor=0.9)
        
        self.play(
            marker.animate.set_color(color_land),
            label_neg1.animate.set_color(color_land),
            Write(label_neg1),
            Write(equation)
        )
        self.wait(2)
