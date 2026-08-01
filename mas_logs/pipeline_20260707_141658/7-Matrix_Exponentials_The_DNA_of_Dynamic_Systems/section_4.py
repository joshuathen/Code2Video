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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial data from storyboard
        title_text = "Visualizing the Transformation (The Vector Field)"
        lecture_lines = [
            "Imagine the system state as a point on a grid.",
            "The matrix exponential acts as a continuous flow operator.",
            "As time passes, the entire grid warps and rotates.",
            "Initial states follow trajectories determined by matrix A.",
            "Watch how the linear transformation evolves over time."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors defined in storyboard
        EMERALD = "#50C878"
        DARK_GREY = "#333333"
        GOLD = "#FFD700"
        
        # 1. Create a 2D coordinate grid
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": BLUE_D, "stroke_opacity": 0.3},
            axis_config={"stroke_color": WHITE, "stroke_opacity": 0.5}
        )

        # Calculate grid center based on A1-F6
        tl = self.grid["A1"]
        br = self.grid["F6"]
        grid_center = np.array([(tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2, 0])

        # 2. Vector Field setup
        # A = [[0, -2], [0.5, 0]]
        def system_func(p):
            local_p = p - grid_center
            x, y = local_p[0], local_p[1]
            dx = -2 * y
            dy = 0.5 * x
            return (RIGHT * dx + UP * dy) * 0.15

        vector_field = ArrowVectorField(
            system_func,
            x_range=[grid_center[0]-2.5, grid_center[0]+2.5, 0.6],
            y_range=[grid_center[1]-2.5, grid_center[1]+2.5, 0.6],
            colors=[DARK_GREY],
            opacity=0.6
        )

        # Group for transformations
        transformation_group = VGroup(plane, vector_field)
        self.place_in_area(transformation_group, 'A1', 'F6', scale_factor=0.85)

        # 3. Create a distinct point (vector) v and label
        v_point = Dot(color=EMERALD, radius=0.08)
        v_label = Text("v(t)", color=EMERALD, font_size=24)
        v_point_group = VGroup(v_point, v_label)
        
        # Anchoring point and label together as per Issue 32
        v_label.add_updater(lambda m: m.next_to(v_point, UR, buff=0.1))
        self.place_at_grid(v_point_group, 'C5', scale_factor=0.7)
        
        # Define the matrix exponential transformation e^{At}
        # e^{At} = [[cos(t), -2*sin(t)], [0.5*sin(t), cos(t)]]
        def get_matrix_at_t(t):
            return np.array([
                [np.cos(t), -2 * np.sin(t), 0],
                [0.5 * np.sin(t), np.cos(t), 0],
                [0, 0, 1]
            ])

        # === Animation for Lecture Line 1 ===
        # Imagine the system state as a point on a grid.
        self.lecture[0].set_color(YELLOW)
        self.play(Create(plane), run_time=1.5)
        self.play(FadeIn(v_point), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The matrix exponential acts as a continuous flow operator.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(vector_field))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # As time passes, the entire grid warps and rotates.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Trajectory setup
        trajectory = TracedPath(v_point.get_center, stroke_color=GOLD, stroke_width=3)
        self.add(trajectory)

        # Save initial states for warping
        plane.save_state()
        start_v_world = v_point.get_center().copy()

        def update_frame(alpha):
            t = alpha * PI # Half period of the elliptical orbit
            mat = get_matrix_at_t(t)
            
            # Warp the grid
            plane.restore()
            plane.apply_matrix(mat, about_point=grid_center)
            
            # Move the vector point
            local_pos = start_v_world - grid_center
            new_local = mat @ local_pos
            v_point.move_to(new_local + grid_center)

        self.play(
            UpdateFromAlphaFunc(plane, lambda m, a: update_frame(a)),
            run_time=6,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Initial states follow trajectories determined by matrix A.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Briefly highlight the trajectory
        self.play(trajectory.animate.set_stroke(width=6), run_time=0.5)
        self.play(trajectory.animate.set_stroke(width=3), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Watch how the linear transformation evolves over time.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Subtle pulse
        pulse = Circle(radius=0.1, color=EMERALD).move_to(v_point)
        self.play(pulse.animate.scale(4).set_opacity(0), run_time=1.5)
        self.remove(pulse)
        self.wait(2)

        # Final cleanup
        self.lecture[4].set_color(WHITE)
