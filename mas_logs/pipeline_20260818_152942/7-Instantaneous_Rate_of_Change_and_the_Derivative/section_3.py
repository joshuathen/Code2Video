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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "As time shrinks, the secant line tilts.",
            "It eventually becomes a tangent line.",
            "This tangent slope shows instantaneous speed.",
            "Point Q slides closer to point P.",
            "The limit defines the exact rate change."
        ]
        self.setup_layout("The Limit Process: Approaching the Tangent", lecture_lines)
        
        # Load assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        self.place_at_grid(ruler, 'A6', scale_factor=0.3)
        self.place_at_grid(protractor, 'F6', scale_factor=0.3)
        self.add(ruler, protractor)
        
        # Create curve and points
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(axes, 'A2', 'E5', scale_factor=0.6)
        curve = axes.plot(lambda x: 0.25 * x**2, x_range=[0, 4], color=WHITE)
        
        p_x = 2.0
        q_x = 3.5
        
        p = Dot(axes.c2p(p_x, 0.25 * p_x**2), color=BLUE)
        q = Dot(axes.c2p(q_x, 0.25 * q_x**2), color=YELLOW)
        
        # Reposition per critic
        self.place_at_grid(p, 'C3', scale_factor=0.7)
        self.place_at_grid(q, 'D4', scale_factor=0.7)
        
        label_p = Text("P", font_size=20).next_to(p, DOWN)
        label_q = Text("Q", font_size=20).next_to(q, UP)
        
        secant = Line(p.get_center(), q.get_center(), color=YELLOW)
        
        self.add(axes, curve, p, q, label_p, label_q, secant)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        q_tracker = ValueTracker(3.5)
        def update_secant(mob):
            # Recalculate based on current dot position
            mob.put_start_and_end_on(p.get_center(), q.get_center())
            
        secant.add_updater(update_secant)
        self.play(q_tracker.animate.set_value(2.2), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        tangent = Line(axes.c2p(1, 0.25), axes.c2p(3, 1.25), color=GREEN) 
        self.place_in_area(tangent, 'A2', 'E5', scale_factor=0.6)
        self.play(FadeIn(tangent), FadeOut(secant))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        self.wait(2)
