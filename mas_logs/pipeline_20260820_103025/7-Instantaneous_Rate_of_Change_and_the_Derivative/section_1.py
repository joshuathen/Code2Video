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
        self.setup_layout("The Problem: Average vs. Instantaneous Speed", [
            "Average speed is total distance over total time.",
            "But what is speed at one specific moment?",
            "Calculus helps us find this instantaneous speed."
        ])
        
        # Setup graph elements
        axes = Axes(x_range=[0, 5], y_range=[0, 5], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.8)
        curve = axes.plot(lambda x: 0.2 * x**3, color=WHITE)
        
        # Assets: Use SVGMobject for .svg files instead of ImageMobject
        cheetah_run = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        
        # Secant line variables
        point_a = axes.c2p(1, 0.2)
        point_b = axes.c2p(4, 3.2)
        secant_line = Line(point_a, point_b, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.place_at_grid(cheetah_run, 'F2', scale_factor=0.5)
        self.play(Create(axes), Create(curve), Create(secant_line), FadeIn(cheetah_run))
        
        avg_speed_label = Text("Average Speed", color=WHITE).scale(0.7)
        self.place_at_grid(avg_speed_label, 'A6', scale_factor=0.6)
        self.place_at_grid(speedometer, 'B6', scale_factor=0.4)
        speedometer.next_to(avg_speed_label, DOWN)
        self.play(FadeIn(avg_speed_label), FadeIn(speedometer))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        self.lecture[0].set_color(WHITE)
        
        point_b_tracker = ValueTracker(4)
        def update_secant(mob):
            x = point_b_tracker.get_value()
            new_b = axes.c2p(x, 0.2 * x**3)
            mob.put_start_and_end_on(point_a, new_b)
            mob.set_color("#FF5733")
            
        secant_line.add_updater(update_secant)
        self.play(point_b_tracker.animate.set_value(1.1), run_time=3)
        secant_line.remove_updater(update_secant)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        self.lecture[1].set_color(WHITE)
        
        tangent_line = Line(axes.c2p(1, 0.2), axes.c2p(2, 0.4), color="#33FF57")
        inst_speed_label = Text("Instantaneous Speed", color="#33FF57").scale(0.7)
        cheetah_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(inst_speed_label, 'D6', scale_factor=0.6)
        self.place_at_grid(cheetah_icon, 'E6', scale_factor=0.4)
        cheetah_icon.next_to(inst_speed_label, DOWN)
        
        self.play(Create(tangent_line), FadeIn(inst_speed_label), FadeIn(cheetah_icon))
        self.play(Flash(tangent_line, color="#33FF57"))
        self.wait(1)
