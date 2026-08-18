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
        self.setup_layout("Visualizing the Secant Line", [
            "A secant line shows average rate of change.",
            "As point B slides to A, slope changes.",
            "The secant line becomes the tangent line."
        ])
        
        # Setup Curve
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, "A3", "F6", scale_factor=0.50)
        
        func = axes.plot(lambda x: 0.25 * x**2, x_range=[0, 4], color=BLUE)
        self.add(axes, func)
        
        # Points and Secant
        x_a = 1
        x_b = 3.5
        
        pt_a = Dot(axes.c2p(x_a, 0.25 * x_a**2), color=YELLOW)
        pt_b = Dot(axes.c2p(x_b, 0.25 * x_b**2), color=RED)
        
        # Adding dummy assets as per requirements (icon/none.svg exists)
        # Assuming placeholders:
        # asset1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        # self.place_at_grid(asset1, "A5", scale_factor=0.7)
        # self.add(asset1)

        line = Line(pt_a.get_center(), pt_b.get_center(), color=WHITE)
        
        self.add(pt_a, pt_b, line)
        self.lecture.set_opacity(1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        line.set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF8C00")
        
        x_b_tracker = ValueTracker(x_b)
        
        def update_line(l):
            new_pt_b = axes.c2p(x_b_tracker.get_value(), 0.25 * x_b_tracker.get_value()**2)
            pt_b.move_to(new_pt_b)
            l.put_start_and_end_on(pt_a.get_center(), pt_b.get_center())
            
        line.add_updater(update_line)
        self.play(x_b_tracker.animate.set_value(1.05), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        line.set_color("#FF4500")
        self.wait(2)
        line.remove_updater(update_line)
