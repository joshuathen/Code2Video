from manim import *
import numpy as np

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
        # Data from storyboard
        title = "The Hook: The Cheetah's Dash"
        lines = [
            "A cheetah sprints, its velocity changing every second.",
            "Differentiation tells us its speed at any moment.",
            "Integration calculates the total distance the cheetah traveled."
        ]
        self.setup_layout(title, lines)
        
        # Colors defined in storyboard
        CHEETAH_COLOR = "#FFFF00"  # Yellow
        GRAPH_COLOR = "#00FF00"    # Green
        SPEED_COLOR = "#FFFFFF"    # White
        ODO_COLOR = "#0000FF"      # Blue
        QM_COLOR = "#FFFFFF"       # White

        # === Animation for Lecture Line 1 ===
        # "A cheetah sprints, its velocity changing every second."
        self.lecture[0].set_color(CHEETAH_COLOR)
        
        # Cheetah silhouette [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg]
        # B034: Self-contained, B058: rate_functions.linear
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg", color=CHEETAH_COLOR)
        self.place_at_grid(cheetah, 'F1', scale_factor=0.6)
        
        # Velocity graph axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 6, 1],
            x_length=4.5,
            y_length=2.5,
            axis_config={"color": WHITE},
            tips=False
        )
        self.place_in_area(axes, 'A1', 'C6', scale_factor=0.8)
        
        # Parabolic-like velocity curve
        # B008: Use Text for alphanumeric labels
        graph = axes.plot(lambda x: 0.1 * x**2 + 0.5 * x, x_range=[0, 5], color=GRAPH_COLOR)
        graph_label = Text("v(t)", color=GRAPH_COLOR, font_size=20)
        self.place_at_grid(graph_label, 'A6')

        # Tracker for cheetah movement and graph tracing
        time_tracker = ValueTracker(0)
        
        # Movement bounds for Cheetah (moving from left to right along row F)
        start_pos = self.grid['F1']
        end_pos = self.grid['F6']
        cheetah.move_to(start_pos)
        
        # Use add_updater for movement
        cheetah.add_updater(lambda m: m.move_to(
            start_pos + (end_pos - start_pos) * (time_tracker.get_value() / 5)
        ))
        
        # Dot on graph tracing velocity
        dot = Dot(color=GRAPH_COLOR)
        dot.add_updater(lambda m: m.move_to(
            axes.c2p(time_tracker.get_value(), 0.1 * time_tracker.get_value()**2 + 0.5 * time_tracker.get_value())
        ))
        
        self.add(axes, graph_label)
        self.play(Create(graph), FadeIn(cheetah), run_time=2)
        self.add(dot)
        self.play(time_tracker.animate.set_value(5), run_time=4, rate_func=rate_functions.linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Differentiation tells us its speed at any moment."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SPEED_COLOR)
        
        # Speedometer icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg]
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg", color=SPEED_COLOR)
        # Issue 21: Position at D2
        self.place_at_grid(speedometer, 'D2', scale_factor=0.8)
        
        self.play(FadeIn(speedometer))
        self.play(Flash(speedometer, color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Integration calculates the total distance the cheetah traveled."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ODO_COLOR)
        
        # Odometer icon (Blue)
        odo_box = Rectangle(width=1.0, height=0.4, color=ODO_COLOR)
        odo_val = Text("123.4", font_size=16, color=ODO_COLOR)
        odometer = VGroup(odo_box, odo_val)
        # Issue 22: Position at D5
        self.place_at_grid(odometer, 'D5', scale_factor=0.8)
        
        # Question mark between icons
        question_mark = Text("?", font_size=48, color=QM_COLOR)
        # Issue 23: Position in area D3-D4
        self.place_in_area(question_mark, 'D3', 'D4', scale_factor=0.8)
        
        self.play(FadeIn(odometer))
        self.wait(0.5)
        self.play(Write(question_mark))
        self.wait(3)
