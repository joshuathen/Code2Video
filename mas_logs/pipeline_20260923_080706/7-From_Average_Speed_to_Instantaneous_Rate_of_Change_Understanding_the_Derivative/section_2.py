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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Shrinking Interval (The Intuitive Limit)", [
            "Move the second point closer to the first.",
            "The secant line becomes a tangent line.",
            "As the interval shrinks, we find instantaneous speed."
        ])
        
        # Setup Graph - reconciled positioning for best readability
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        self.add(axes)
        
        curve = axes.plot(lambda x: 0.25 * x**2, x_range=[0, 4], color=BLUE)
        self.add(curve)
        
        p1 = Dot(axes.c2p(1, 0.25), color=WHITE)
        p2 = Dot(axes.c2p(3, 2.25), color="#32CD32")
        
        secant = Line(p1.get_center(), p2.get_center(), color=YELLOW)
        
        # Load Assets
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        self.place_at_grid(speedometer, 'A4', scale_factor=0.3)
        self.add(speedometer)
        
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        self.place_at_grid(stopwatch, 'A6', scale_factor=0.3)
        self.add(stopwatch)
        
        self.add(p1, p2, secant)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#32CD32")
        
        t = ValueTracker(3.0)
        
        secant.add_updater(lambda mob: mob.put_start_and_end_on(
            p1.get_center(),
            axes.c2p(t.get_value(), 0.25 * t.get_value()**2)
        ))
        
        p2.add_updater(lambda mob: mob.move_to(
            axes.c2p(t.get_value(), 0.25 * t.get_value()**2)
        ))
        
        self.play(t.animate.set_value(1.2), run_time=3)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(t.animate.set_value(1.01), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.play(t.animate.set_value(1.001), run_time=1)
        self.wait(0.5)
        
        secant.remove_updater(secant.get_updaters()[0])
        p2.remove_updater(p2.get_updaters()[0])
