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
        self.setup_layout("Prerequisite Warm-up: The Concept of Change", [
            "Derivative represents the rate of change.",
            "Visualizing functions as moving, changing parts.",
            "The runner changes based on combined functions."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create a central point #FFFFFF labeled 'f(x)' using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/runner.svg].
        runner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/runner.svg", color=WHITE)
        label = MathTex("f(x)", color=WHITE).next_to(runner, UP)
        group = VGroup(runner, label)
        self.place_at_grid(group, "C4")
        self.play(FadeIn(runner), Write(label))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Expand the point into a dynamic curve using #3498DB color.
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2, color="#3498DB")
        anim_group = VGroup(axes, curve)
        
        # Using recommended grid area C3-F6
        self.place_in_area(anim_group, 'C3', 'F6', scale_factor=0.6)
        
        self.play(
            FadeOut(group),
            Create(axes), Create(curve)
        )
        self.lecture[1].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the tangent line slope #E74C3C moving along curve
        x_val = ValueTracker(-1.5)
        
        # Helper to get tangent line
        tangent = Line(start=LEFT, end=RIGHT, color="#E74C3C")
        
        def update_tangent(mob):
            x = x_val.get_value()
            slope = 2 * x
            mob.set_angle(np.arctan(slope))
            mob.move_to(axes.c2p(x, x**2))
            
        tangent.add_updater(update_tangent)
        self.add(tangent)
        
        self.play(x_val.animate.set_value(1.5), run_time=3, rate_func=linear)
        self.lecture[2].set_color("#E74C3C")
        self.wait(1)
        
        # Clean up
        tangent.remove_updater(update_tangent)
