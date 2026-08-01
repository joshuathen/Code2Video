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
        # Setup layout based on storyboard
        title_text = "The Prior and the Likelihood (The Slicing)"
        lecture_lines = [
            "We start with the probability of rain.",
            "This 'Prior' is represented by a vertical column.",
            "Next, we consider how often evidence occurs.",
            "A horizontal slice represents the likelihood of a beep.",
            "This creates specific regions for rain and no rain."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Establish the 1x1 probability square
        square_size = 3.0
        rain_width = square_size * 0.2
        no_rain_width = square_size * 0.8
        
        # Main square border
        sq_border = Rectangle(width=square_size, height=square_size, color="#FFFFFF", stroke_width=2)
        
        # Rain column (Prior) - Blue
        rain_col = Rectangle(
            width=rain_width, height=square_size, 
            fill_color="#0000FF", fill_opacity=0.3, stroke_width=1
        )
        rain_col.align_to(sq_border, LEFT)
        
        # No Rain column - Gray
        no_rain_col = Rectangle(
            width=no_rain_width, height=square_size, 
            fill_color="#888888", fill_opacity=0.1, stroke_width=1
        )
        no_rain_col.align_to(sq_border, RIGHT)
        
        # Group and place
        # [Issue 32 Fix]: Positioned at C2-F5 to avoid crowding Row A/B
        prob_space = VGroup(sq_border, rain_col, no_rain_col)
        self.place_in_area(prob_space, "C2", "F5", scale_factor=1.0) 

        self.lecture[0].set_color("#0000FF")
        self.play(Create(sq_border), run_time=1.0)
        self.play(FadeIn(rain_col), FadeIn(no_rain_col))
        self.wait(1.0)

        # === Animation for Lecture Line 2 ===
        # Prior Label
        prior_label = Text("P(Rain) = 0.2", font_size=20, color="#0000FF")
        # Position label in Row B, above the rain column (Column 2)
        self.place_at_grid(prior_label, "B2", scale_factor=0.8)
        
        self.lecture[1].set_color("#0000FF")
        self.play(Write(prior_label))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Pause to transition to evidence
        self.wait(1.0)

        # === Animation for Lecture Line 4 ===
        # Horizontal slices for likelihood: P(Beep|Rain)=0.9, P(Beep|NoRain)=0.1
        tp_height = square_size * 0.9
        fp_height = square_size * 0.1
        
        tp_slice = Line(
            start=rain_col.get_bottom() + UP * tp_height + LEFT * (rain_width / 2),
            end=rain_col.get_bottom() + UP * tp_height + RIGHT * (rain_width / 2),
            color="#FFD700", stroke_width=4
        )
        fp_slice = Line(
            start=no_rain_col.get_bottom() + UP * fp_height + LEFT * (no_rain_width / 2),
            end=no_rain_col.get_bottom() + UP * fp_height + RIGHT * (no_rain_width / 2),
            color="#FFD700", stroke_width=4
        )
        
        # Label the horizontal slices
        # [Issue 31 Fix]: Using place_in_area for multi-word formula label
        likelihood_label = Text("Likelihood: P(Evidence | Event)", font_size=20, color="#FFFFFF")
        self.place_in_area(likelihood_label, "A2", "A5", scale_factor=0.7)
        
        self.lecture[3].set_color("#FFD700")
        self.play(Create(tp_slice), Create(fp_slice))
        self.play(Write(likelihood_label))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Highlighted regions (Intersections)
        # Rain and Beep (True Positive) - Green
        rain_beep_rect = Rectangle(
            width=rain_width, height=tp_height,
            fill_color="#00FF00", fill_opacity=0.5, stroke_width=0
        )
        rain_beep_rect.align_to(rain_col, DOWN).align_to(rain_col, LEFT)
        
        # No Rain and Beep (False Positive) - Red
        no_rain_beep_rect = Rectangle(
            width=no_rain_width, height=fp_height,
            fill_color="#FF0000", fill_opacity=0.5, stroke_width=0
        )
        no_rain_beep_rect.align_to(no_rain_col, DOWN).align_to(no_rain_col, LEFT)
        
        # Region labels
        tp_tag = Text("True Pos", font_size=16, color="#00FF00")
        fp_tag = Text("False Pos", font_size=16, color="#FF0000")
        
        # Proximity placement (L002) below the rectangles
        tp_tag.scale(0.8).next_to(rain_beep_rect, DOWN, buff=0.1)
        fp_tag.scale(0.8).next_to(no_rain_beep_rect, DOWN, buff=0.1)

        self.lecture[4].set_color("#00FF00")
        self.play(FadeIn(rain_beep_rect), FadeIn(no_rain_beep_rect))
        self.play(Write(tp_tag), Write(fp_tag))
        self.wait(3.0)
