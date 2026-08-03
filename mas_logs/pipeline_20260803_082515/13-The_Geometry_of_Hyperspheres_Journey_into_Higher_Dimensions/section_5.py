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
        # Data from storyboard
        title_text = "Where the Surface Is: All Crust, No Bread"
        lecture_lines = [
            "Most volume concentrates near the hypersphere's surface.",
            "Even a thin peel contains nearly all the fruit.",
            "High-dimensional spheres are essentially just empty shells."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # A large sphere with a thick orange (#FFA500) outer shell.
        # We use Orange for the relevant lecture line.
        self.lecture[0].set_color("#FFA500")
        
        # Outer shell (orange) and inner fruit (white)
        outer_circle = Circle(radius=2, color="#FFA500", fill_opacity=1, stroke_width=0)
        inner_circle = Circle(radius=1.8, color="#FFFFFF", fill_opacity=1, stroke_width=0)
        
        sphere_viz = VGroup(outer_circle, inner_circle)
        # Center the sphere in the main grid area (B2-E5)
        self.place_in_area(sphere_viz, "B2", "E5", scale_factor=0.8)
        
        self.play(FadeIn(outer_circle), FadeIn(inner_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The inner white region (#FFFFFF) shrinks rapidly as a 'Dimension' counter increases.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF") 

        dim_tracker = ValueTracker(3)
        dim_label = Text("Dimension: ", font_size=24, color=WHITE)
        dim_number = DecimalNumber(3, num_decimal_places=0, font_size=24, color=WHITE)
        counter_group = VGroup(dim_label, dim_number).arrange(RIGHT)
        
        # Resolve Issue 30: Place in area A3-A4 for vertical alignment with sphere
        self.place_in_area(counter_group, 'A3', 'A4', scale_factor=1.0)

        # Updater for the number
        dim_number.add_updater(lambda d: d.set_value(dim_tracker.get_value()))
        
        # Initial state for radius scaling
        initial_inner_radius = inner_circle.width / 2
        
        def update_inner_circle(m):
            d = dim_tracker.get_value()
            # Visual representation of volume ratio: (r/R)^d
            # In 2D viz, we scale radius such that Area/Area_total represents volume ratio
            # r_viz = R_viz * (0.9 ** (d/2)) - assuming crust is 10% thickness in higher dims
            # For d=3, we anchor the current size as the starting point.
            current_scale = (0.9 ** ((d - 3) / 2))
            new_width = max(0.05, 2 * initial_inner_radius * current_scale)
            m.set_width(new_width)
            m.move_to(outer_circle.get_center())

        inner_circle.add_updater(update_inner_circle)

        self.play(Write(counter_group))
        # Animate dimension from 3 to 100 rapidly
        self.play(dim_tracker.animate.set_value(100), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The sphere is now almost entirely orange crust, labeled '99% Volume'.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFA500")

        volume_label = Text("99% Volume", color="#FFA500", font_size=24)
        
        # Resolve Issue 31: Place in area F3-F4 for vertical alignment with sphere
        self.place_in_area(volume_label, 'F3', 'F4', scale_factor=1.0)
        
        # Arrow pointing to the orange shell from below
        arrow = Arrow(
            start=volume_label.get_top(), 
            end=outer_circle.get_bottom() + UP * 0.1, 
            color="#FFA500", 
            stroke_width=4,
            buff=0.1
        )
        
        self.play(
            FadeIn(volume_label),
            Create(arrow)
        )
        self.wait(2)
