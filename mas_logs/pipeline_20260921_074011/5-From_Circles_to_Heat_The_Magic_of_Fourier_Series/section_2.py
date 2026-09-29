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
        lecture_lines = ["Euler's formula connects circles to complex waves.", "Rotation projected creates simple sine waves.", "These waves form the basis of signals."]
        self.setup_layout("The Bridge: From Circles to Sine/Cosine Waves", lecture_lines)
        
        # Assets
        antenna = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg")
        radio = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radio.svg")
        
        # Elements
        circle = Circle(radius=0.8, color=WHITE)
        self.place_in_area(circle, 'B2', 'C3', scale_factor=0.6)
        self.place_at_grid(antenna, 'B1', scale_factor=0.3)
        
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda m: m.move_to(circle.get_center() + 0.8 * RIGHT * np.cos(tracker.get_value()) + 0.8 * UP * np.sin(tracker.get_value())))
        
        tracker = ValueTracker(0)
        
        graph = Axes(x_range=[0, 4, 1], y_range=[-1, 1, 1], axis_config={"include_tip": False}, x_length=2.5, y_length=1.5)
        self.place_at_grid(graph, 'D4', scale_factor=0.65)
        self.place_at_grid(radio, 'D6', scale_factor=0.3)
        
        # Persistent mobjects
        sine_curve = VMobject(color="#00FFFF")
        def update_curve(mob):
            t = tracker.get_value()
            new_curve = graph.plot(lambda x: 0.8 * np.sin(x), x_range=[0, min(t, 4)], color="#00FFFF")
            mob.become(new_curve)
        sine_curve.add_updater(update_curve)
        
        dashed_line = DashedLine(start=dot.get_center(), end=dot.get_center())
        def update_line(mob):
            mob.put_start_and_end_on(dot.get_center(), graph.c2p(0, 0.8 * np.sin(tracker.get_value())))
        dashed_line.add_updater(update_line)
        
        self.add(circle, dot, graph, sine_curve, dashed_line, antenna, radio)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(tracker.animate.set_value(4 * PI), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#90EE90"))
        self.wait(2)
