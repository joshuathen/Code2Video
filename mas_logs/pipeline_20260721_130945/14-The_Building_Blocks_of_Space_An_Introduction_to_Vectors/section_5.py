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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and Lecture lines retrieval
        title_text = "Real-World Application: Computer Graphics"
        lecture_lines = [
            "Vectors power movement in modern video games.",
            "Slingshots use vectors to calculate speed and angle.",
            "These building blocks make digital worlds come alive."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Dim all lecture lines initially
        for line in self.lecture:
            line.set_color(GRAY_D)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line (white)
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create a brown slingshot (#8B4513) using procedural shapes
        slingshot_base = Line(ORIGIN, DOWN * 1.0, color="#8B4513", stroke_width=12)
        slingshot_left = Line(ORIGIN, UP * 0.6 + LEFT * 0.4, color="#8B4513", stroke_width=10)
        slingshot_right = Line(ORIGIN, UP * 0.6 + RIGHT * 0.4, color="#8B4513", stroke_width=10)
        slingshot = VGroup(slingshot_base, slingshot_left, slingshot_right)
        
        # Position slingshot at D4 with scale 0.8 to resolve Issue 32 and 33
        # This prevents the pullback animation from crowding the lecture area and fits better in the grid.
        self.place_at_grid(slingshot, "D4", scale_factor=0.8)
        
        # Launch point is between the slingshot forks
        launch_point = slingshot.get_critical_point(UP) + DOWN * 0.1
        
        # White vector arrow (#FFFFFF) representing initial pull
        pull_vector = Arrow(
            launch_point, 
            launch_point + LEFT * 0.8 + DOWN * 0.3, 
            color=WHITE, 
            buff=0,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.2
        )
        
        self.play(
            Create(slingshot),
            GrowArrow(pull_vector),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(GRAY_D),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # ValueTrackers for dynamic stretching and rotation
        offset_x = ValueTracker(-0.8)
        offset_y = ValueTracker(-0.3)
        
        # Updater to make the arrow dynamic (rotates and stretches)
        pull_vector.add_updater(
            lambda m: m.become(
                Arrow(
                    launch_point,
                    launch_point + np.array([offset_x.get_value(), offset_y.get_value(), 0]),
                    color=WHITE,
                    buff=0,
                    stroke_width=6,
                    max_tip_length_to_length_ratio=0.2
                )
            )
        )
        
        # Animate the pull vector rotating and stretching dynamically
        self.play(
            offset_x.animate.set_value(-1.6),
            offset_y.animate.set_value(-0.8),
            run_time=1.2
        )
        self.play(
            offset_x.animate.set_value(-0.4),
            offset_y.animate.set_value(0.4),
            run_time=1.2
        )
        self.play(
            offset_x.animate.set_value(-1.2),
            offset_y.animate.set_value(-0.2),
            run_time=1.0
        )
        
        pull_vector.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line with Green to match trajectory
        self.play(
            self.lecture[1].animate.set_color(GRAY_D),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        # A green dashed trajectory (#00FF00) extends from the slingshot
        def trajectory_path(t):
            v_x = 2.0
            v_y = 1.5
            gravity = 1.8
            x = v_x * t
            y = v_y * t - 0.5 * gravity * t**2
            return launch_point + np.array([x, y, 0])
            
        trajectory_mobject = ParametricFunction(
            trajectory_path,
            t_range=[0, 1.8],
            color="#00FF00"
        )
        
        dashed_trajectory = DashedVMobject(trajectory_mobject, num_dashes=30)
        
        self.play(
            Create(dashed_trajectory),
            run_time=2.5
        )
        
        self.play(FadeOut(pull_vector), run_time=0.5)
        self.wait(2)
