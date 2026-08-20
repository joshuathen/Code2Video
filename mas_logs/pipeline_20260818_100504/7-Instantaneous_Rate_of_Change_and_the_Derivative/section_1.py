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
        self.setup_layout("The Problem of Speed", [
            "Average speed is total distance over total time.",
            "But averages hide speed at specific moments.",
            "We need speed at an exact instant."
        ])
        
        # Define curve and points
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.25 * x**3, color=WHITE)
        
        # Layout according to Critic suggestions
        self.place_in_area(axes, 'C2', 'E5', scale_factor=0.5)
        
        # Assets
        tachometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tachometer.svg")
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        self.place_at_grid(tachometer, 'A2', scale_factor=0.3)
        self.place_at_grid(stopwatch, 'F5', scale_factor=0.3)
        
        self.add(curve, tachometer, stopwatch)

        p = Dot(axes.c2p(1, 0.25), color=WHITE)
        q = Dot(axes.c2p(3, 2.25), color=WHITE)
        self.add(p, q)

        # === Animation for Lecture Line 1 ===
        # Average speed is total distance over total time.
        self.lecture[0].set_color("#FFFFFF")
        secant = Line(p.get_center(), q.get_center(), color="#FF0000")
        self.add(secant)

        # === Animation for Lecture Line 2 ===
        # But averages hide speed at specific moments.
        self.lecture[1].set_color("#00FF00")
        
        # Update Q position using ValueTracker
        t = ValueTracker(3)
        def update_q(mob):
            x = t.get_value()
            mob.move_to(axes.c2p(x, 0.25 * x**3))
        
        def update_secant(mob):
            mob.put_start_and_end_on(p.get_center(), q.get_center())
            
        q.add_updater(update_q)
        secant.add_updater(update_secant)
        
        self.play(t.animate.set_value(1.5), run_time=2)

        # === Animation for Lecture Line 3 ===
        # We need speed at an exact instant.
        self.lecture[2].set_color("#FF00FF")
        
        tangent = Line(color="#FFFF00")
        tangent.add_updater(lambda mob: mob.put_start_and_end_on(
            axes.c2p(1 - 0.5, 0.25 * 1**3 - 0.375), 
            axes.c2p(1 + 0.5, 0.25 * 1**3 + 0.375)
        ))
        
        self.play(FadeIn(tangent), run_time=1)
        self.play(t.animate.set_value(1.0), run_time=2)
