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
        # Section Title and Lecture Lines
        title_text = "Scalar Multiplication: Scaling the World"
        lecture_lines = [
            "Scalars are regular numbers that scale our vectors.",
            "Multiplying by a scalar stretches or shrinks the arrow.",
            "A scalar of two doubles the vector's length.",
            "Negative scalars flip the vector to the opposite direction.",
            "This scaling is the core of linear transformations."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color Constants
        VECTOR_COLOR = "#ADFF2F"  # Greenish-yellow as requested
        SCALAR_COLOR = "#FFD700"  # Gold for scalars
        
        # === Animation for Lecture Line 1 ===
        # "Scalars are regular numbers that scale our vectors."
        self.play(self.lecture[0].animate.set_color(SCALAR_COLOR))
        
        # Asset integration: World icon in the background
        world_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/world.svg")
        world_icon.set_opacity(0.15)
        self.place_in_area(world_icon, "B3", "F6", scale_factor=2.5)
        
        # Mathematical representation of the initial vector
        vec_v_math = MathTex(r"\vec{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", color=VECTOR_COLOR)
        self.place_at_grid(vec_v_math, "A3", scale_factor=0.8)
        
        # Create a coordinate system (NumberPlane) to visualize vectors
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-7, 7, 1],
            x_length=3.5,
            y_length=4.5,
            axis_config={"include_tip": True, "stroke_width": 1.5, "color": GREY},
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, "B3", "F6")
        
        # The vector mobject
        v_arrow = plane.get_vector([1, 2], color=VECTOR_COLOR)
        v_label = MathTex(r"\vec{v}", color=VECTOR_COLOR)
        
        # Label updater to stay near the arrow tip
        def update_label(m):
            end_pos = v_arrow.get_end()
            # Dynamic placement to avoid overlapping the origin/axis
            direction = UR if end_pos[1] >= plane.get_origin()[1] else DL
            m.next_to(end_pos, direction, buff=0.1)
        
        v_label.add_updater(update_label)

        self.play(FadeIn(world_icon))
        self.play(Write(vec_v_math))
        self.play(Create(plane))
        self.play(GrowArrow(v_arrow), FadeIn(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Multiplying by a scalar stretches or shrinks the arrow."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(SCALAR_COLOR)
        )
        
        # Scalar tracker for smooth animation
        k_tracker = ValueTracker(1.0)
        
        # Issue #30: Relocate the scalar tracker to A6
        scalar_label = MathTex("k =", color=SCALAR_COLOR)
        scalar_val = DecimalNumber(1.0, color=SCALAR_COLOR, num_decimal_places=1)
        scalar_val.add_updater(lambda m: m.set_value(k_tracker.get_value()))
        scalar_group = VGroup(scalar_label, scalar_val).arrange(RIGHT, buff=0.2)
        self.place_at_grid(scalar_group, "A6", scale_factor=0.7)
        
        # Issue #29: Use place_in_area for A4-A5 for result expression
        res_math = MathTex(r"k \cdot \vec{v}", color=WHITE)
        self.place_in_area(res_math, "A4", "A5", scale_factor=0.8)
        
        # Updater to scale the vector arrow geometry based on k
        def update_arrow(m):
            k = k_tracker.get_value()
            new_arrow = plane.get_vector([k * 1, k * 2], color=VECTOR_COLOR)
            m.become(new_arrow)
            
        v_arrow.add_updater(update_arrow)
        
        self.play(FadeIn(scalar_group), Write(res_math))
        
        # Pulse the vector as it starts scaling (transitioning to Line 2 action)
        self.play(v_arrow.animate.set_stroke(width=10), run_time=0.3)
        self.play(v_arrow.animate.set_stroke(width=6), run_time=0.3)
        
        # Small demo of scaling
        self.play(k_tracker.animate.set_value(1.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A scalar of two doubles the vector's length."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SCALAR_COLOR)
        )
        
        # Animate scalar growing to 2
        self.play(k_tracker.animate.set_value(2.0), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Negative scalars flip the vector to the opposite direction."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(SCALAR_COLOR)
        )
        
        # Animate scalar flipping to -1
        self.play(k_tracker.animate.set_value(-1.0), run_time=2.0)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This scaling is the core of linear transformations."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(VECTOR_COLOR)
        )
        
        # Animate scalar to 3.0 (final example)
        self.play(k_tracker.animate.set_value(3.0), run_time=1.5)
        
        # Issue #29: Final result mathematical formula in A4-A5
        final_res_math = MathTex(r"3 \cdot \vec{v} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}", color=VECTOR_COLOR)
        self.place_in_area(final_res_math, "A4", "A5", scale_factor=0.8)
        
        self.play(ReplacementTransform(res_math, final_res_math))
        
        # Pulse the final result as per storyboard "Pulse the scaled #ADFF2F vector"
        self.play(v_arrow.animate.scale(1.2), run_time=0.2, rate_func=there_and_back)
        self.play(v_arrow.animate.scale(1.2), run_time=0.2, rate_func=there_and_back)
        
        self.wait(2)

        # Cleanup updaters to avoid runtime overhead
        v_arrow.remove_updater(update_arrow)
        v_label.remove_updater(update_label)
        scalar_val.clear_updaters()
