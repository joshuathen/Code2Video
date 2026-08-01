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
        # Setup basic layout
        lines = [
            'CLIP uses two encoders to process text and images.',
            'These encoders project data into high-dimensional vectors.',
            'We measure alignment using the cosine similarity formula.',
            'High dot products mean the text describes the image.',
            'Vectors rotate together as the description matches the visual.'
        ]
        self.setup_layout("CLIP: The Translator’s Compass", lines)
        
        # Colors
        COLOR_TEXT = "#5555FF"
        COLOR_IMAGE = "#FF5555"
        COLOR_FORMULA = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_TEXT)
        
        text_box = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.8, width=2.0, color=COLOR_TEXT),
            Text("Text Encoder", font_size=18, color=COLOR_TEXT)
        )
        self.place_at_grid(text_box, "A2")
        
        image_box = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.8, width=2.0, color=COLOR_IMAGE),
            Text("Image Encoder", font_size=18, color=COLOR_IMAGE)
        )
        self.place_at_grid(image_box, "A5")
        
        joint_space = Circle(radius=1.5, color=WHITE, stroke_width=2).set_opacity(0.2)
        self.place_in_area(joint_space, "C2", "E5")
        
        arrow_text = Arrow(text_box.get_bottom(), joint_space.get_top() + LEFT*0.5, color=COLOR_TEXT, buff=0.1)
        arrow_image = Arrow(image_box.get_bottom(), joint_space.get_top() + RIGHT*0.5, color=COLOR_IMAGE, buff=0.1)
        
        self.play(
            FadeIn(text_box),
            FadeIn(image_box),
            FadeIn(joint_space),
            Create(arrow_text),
            Create(arrow_image),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_IMAGE)
        
        center_pt = joint_space.get_center()
        
        # Vector t (Text)
        vec_t = Arrow(center_pt, center_pt + UP*1.2, color=COLOR_TEXT, buff=0)
        label_t = Text("A fluffy Samoyed", font_size=14, color=COLOR_TEXT)
        label_t.next_to(vec_t.get_end(), UP, buff=0.1)
        
        # Vector i (Image) - Initially at a wide angle
        vec_i = Arrow(center_pt, center_pt + rotate_vector(UP*1.2, -120*DEGREES), color=COLOR_IMAGE, buff=0)
        label_i = Text("Noisy Image", font_size=14, color=COLOR_IMAGE)
        label_i.next_to(vec_i.get_end(), DOWN+LEFT, buff=0.1)
        
        self.play(
            GrowArrow(vec_t),
            FadeIn(label_t),
            run_time=1
        )
        self.play(
            GrowArrow(vec_i),
            FadeIn(label_i),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_FORMULA)
        
        # Formula: sim(t, i) = cos(theta)
        # Using unicode for theta
        formula = Text("sim(t, i) = cos(θ)", font_size=24, color=COLOR_FORMULA)
        # Fix: Issue 39 & 55 - Use place_in_area to avoid crowding and edge proximity
        self.place_in_area(formula, 'F4', 'F6', scale_factor=0.7)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_FORMULA)
        
        # Just highlighting the conceptual link
        dot_product_text = Text("t · i = ||t|| ||i|| cos(θ)", font_size=18, color=COLOR_FORMULA)
        # Position dot product above formula in the same horizontal span
        self.place_in_area(dot_product_text, "E4", "E6", scale_factor=0.7)
        
        self.play(FadeIn(dot_product_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        
        # Rotation of vector i to align with t
        new_vec_i = Arrow(center_pt, center_pt + UP*1.2, color=COLOR_IMAGE, buff=0)
        new_label_i = Text("Samoyed Image", font_size=14, color=COLOR_IMAGE)
        new_label_i.next_to(new_vec_i.get_end(), RIGHT, buff=0.1)
        
        # Similarity score indicator (conceptual)
        sim_val = Text("Similarity: 0.2 -> 0.95", font_size=16, color=WHITE)
        # Fix: Issue 38 & 55 - Move to F3 and scale to 0.8 to reduce crowding with lecture notes
        self.place_at_grid(sim_val, 'F3', scale_factor=0.8)
        
        self.play(
            ReplacementTransform(vec_i, new_vec_i),
            ReplacementTransform(label_i, new_label_i),
            FadeIn(sim_val),
            run_time=2
        )
        
        self.play(
            new_vec_i.animate.set_color(YELLOW),
            vec_t.animate.set_color(YELLOW),
            run_time=0.5
        )
        self.wait(2)
