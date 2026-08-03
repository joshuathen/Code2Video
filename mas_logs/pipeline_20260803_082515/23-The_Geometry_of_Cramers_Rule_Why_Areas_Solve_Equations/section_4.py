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
        # Title and Lecture Lines
        lines = [
            "Replace one column with the target vector.",
            "This creates a new, sliding parallelogram area.",
            "Adding a multiple of another vector shears it.",
            "Shearing doesn't change the area's base or height.",
            "So, the new area is x times the original."
        ]
        self.setup_layout("The Geometric Trick: The 'Sliding' Area", lines)

        # Colors
        COLOR_V1V2 = "#00FFFF" # Cyan
        COLOR_B = "#FF00FF"    # Magenta
        COLOR_TEXT = "#FFFFFF" # White
        COLOR_DECOMP = "#FFFF00" # Yellow
        COLOR_XV1 = "#FFA500" # Orange for x*v1

        # Define vectors based on x*v1 + y*v2 = b
        # Let x=2, y=1, v1=[1.2, 0], v2=[0.5, 1.2]
        v1_val = np.array([1.2, 0, 0])
        v2_val = np.array([0.5, 1.2, 0])
        x_val, y_val = 2.0, 1.0
        b_val = x_val * v1_val + y_val * v2_val
        
        # Scale for grid visualization
        plot_scale = 1.0
        
        def to_plot(vec):
            return vec * plot_scale

        # Origin point on the grid (bottom area)
        plot_origin = self.grid["F2"]
        
        def get_pos(vec):
            return plot_origin + to_plot(vec)

        # === Animation for Lecture Line 1 ===
        # Replace one column with the target vector.
        self.lecture[0].set_color(YELLOW)
        
        v1_arr = Arrow(plot_origin, get_pos(v1_val), buff=0, color=COLOR_V1V2, stroke_width=4)
        v2_arr = Arrow(plot_origin, get_pos(v2_val), buff=0, color=COLOR_V1V2, stroke_width=4)
        v1_lab = MathTex("v_1", color=COLOR_V1V2, font_size=24).next_to(v1_arr.get_end(), DOWN, buff=0.1)
        v2_lab = MathTex("v_2", color=COLOR_V1V2, font_size=24).next_to(v2_arr.get_end(), LEFT, buff=0.1)
        
        # Original Parallelogram (v1, v2)
        p_orig = Polygon(
            plot_origin, get_pos(v1_val), get_pos(v1_val + v2_val), get_pos(v2_val),
            stroke_width=2, stroke_color=COLOR_V1V2, fill_color=COLOR_V1V2, fill_opacity=0.3
        )
        
        self.play(Create(v1_arr), Create(v2_arr), Write(v1_lab), Write(v2_lab))
        self.play(Create(p_orig))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This creates a new, sliding parallelogram area.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        b_arr = Arrow(plot_origin, get_pos(b_val), buff=0, color=COLOR_B, stroke_width=4)
        b_lab = MathTex("b", color=COLOR_B, font_size=24).next_to(b_arr.get_end(), RIGHT, buff=0.1)
        
        # New Parallelogram (b, v2)
        p_new = Polygon(
            plot_origin, get_pos(b_val), get_pos(b_val + v2_val), get_pos(v2_val),
            stroke_width=2, stroke_color=COLOR_B, fill_color=COLOR_B, fill_opacity=0.3
        )
        
        self.play(Create(b_arr), Write(b_lab))
        self.play(FadeIn(p_new))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adding a multiple of another vector shears it.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Show b = x*v1 + y*v2
        b_eq = MathTex("b = x v_1 + y v_2", color=COLOR_TEXT, font_size=32)
        # Resolved Issue #41: Positioning b_eq at A6
        self.place_at_grid(b_eq, "A6", scale_factor=0.7)
        
        x_v1_vec = x_val * v1_val
        y_v2_vec = y_val * v2_val
        
        # Visual decomposition
        x_v1_arr = Arrow(plot_origin, get_pos(x_v1_vec), buff=0, color=COLOR_XV1, stroke_width=3)
        y_v2_arr = Arrow(get_pos(x_v1_vec), get_pos(b_val), buff=0, color=COLOR_DECOMP, stroke_width=3)
        
        x_v1_lab = MathTex("x v_1", color=COLOR_XV1, font_size=22).next_to(x_v1_arr.get_center(), DOWN, buff=0.1)
        y_v2_lab = MathTex("y v_2", color=COLOR_DECOMP, font_size=22).next_to(y_v2_arr.get_center(), RIGHT, buff=0.1)
        
        self.play(Write(b_eq))
        self.play(Create(x_v1_arr), Create(y_v2_arr), Write(x_v1_lab), Write(y_v2_lab))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Shearing doesn't change the area's base or height.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Sliding mechanism using a ValueTracker
        slide_tracker = ValueTracker(0)
        
        # The parallelogram that updates its shape
        moving_p = p_new.copy()
        
        def update_p(p):
            t = slide_tracker.get_value()
            # Slide from b towards x*v1
            current_tip = b_val - t * y_v2_vec
            p.set_points_as_corners([
                plot_origin, 
                get_pos(current_tip), 
                get_pos(current_tip + v2_val), 
                get_pos(v2_val),
                plot_origin
            ])
            
        moving_p.add_updater(update_p)
        self.add(moving_p)
        self.remove(p_new)
        
        # Slide along v2 direction
        guide_line = DashedLine(get_pos(b_val), get_pos(x_v1_vec), color=WHITE, stroke_width=1)
        self.play(Create(guide_line))
        self.play(slide_tracker.animate.set_value(1), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # So, the new area is x times the original.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_eq = MathTex(
            "Area(b, v_2) = x \\cdot Area(v_1, v_2)",
            color=COLOR_TEXT, font_size=32
        )
        # Resolved Issue #40: Positioning final_eq at B5-B6
        self.place_in_area(final_eq, "B5", "B6", scale_factor=0.6)
        
        # Show two copies of original area (since x=2)
        p_chunk1 = p_orig.copy().set_fill(opacity=0.6).set_stroke(width=4)
        p_chunk2 = Polygon(
            get_pos(v1_val), get_pos(2*v1_val), get_pos(2*v1_val + v2_val), get_pos(v1_val + v2_val),
            stroke_width=4, stroke_color=COLOR_V1V2, fill_color=COLOR_V1V2, fill_opacity=0.6
        )
        
        self.play(Write(final_eq))
        self.play(FadeIn(p_chunk1))
        self.play(FadeIn(p_chunk2))
        self.play(Indicate(final_eq))
        self.wait(2)
