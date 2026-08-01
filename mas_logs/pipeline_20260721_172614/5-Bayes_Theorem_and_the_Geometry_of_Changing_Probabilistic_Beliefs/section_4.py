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
        # Define lecture lines from storyboard
        title = "The Geometric Shift: Shrinking the Universe"
        lecture_lines = [
            "Observing evidence eliminates all 'no mound' possibilities.",
            "The remaining areas represent our narrowed-down reality.",
            "Now, the total probability space has significantly shrunk.",
            "We re-scale these remnants back into a full square.",
            "This physical transformation represents our updated belief."
        ]
        self.setup_layout(title, lecture_lines)

        # Proportions
        p_bone = 0.4
        p_m_b = 0.8
        p_m_nb = 0.2
        
        # Colors
        color_a = "#5dade2" # Bone + Mound (Light Blue)
        color_b = "#aed6f1" # Bone + No Mound (Faded Blue)
        color_c = "#f5b041" # No Bone + Mound (Light Orange)
        color_d = "#fad7a0" # No Bone + No Mound (Faded Orange)
        color_gray = "#A9A9A9"

        # Square size derived from area B2 to E5
        sq_size = 3.0
        square_outline = Rectangle(width=sq_size, height=sq_size, color=WHITE, stroke_width=2)
        self.place_in_area(square_outline, "B2", "E5")
        
        # Internal Rectangles
        w_b = p_bone * sq_size
        w_nb = (1 - p_bone) * sq_size
        
        # Left side (Bone)
        rect_a = Rectangle(width=w_b, height=p_m_b * sq_size, fill_opacity=0.8, fill_color=color_a, stroke_width=1)
        rect_b = Rectangle(width=w_b, height=(1-p_m_b) * sq_size, fill_opacity=0.8, fill_color=color_b, stroke_width=1)
        bone_col = VGroup(rect_a, rect_b).arrange(DOWN, buff=0)
        
        # Right side (No Bone)
        rect_c = Rectangle(width=w_nb, height=p_m_nb * sq_size, fill_opacity=0.8, fill_color=color_c, stroke_width=1)
        rect_d = Rectangle(width=w_nb, height=(1-p_m_nb) * sq_size, fill_opacity=0.8, fill_color=color_d, stroke_width=1)
        nobone_col = VGroup(rect_c, rect_d).arrange(DOWN, buff=0)
        
        full_square_content = VGroup(bone_col, nobone_col).arrange(RIGHT, buff=0)
        full_square_content.move_to(square_outline.get_center())

        # Labels - Issues 36 and 38
        label_bone_mound = Text("Bone & Mound", font_size=24)
        self.place_at_grid(label_bone_mound, 'B2', scale_factor=0.4)
        
        label_bone_no_mound = Text("Bone & No Mound", font_size=24)
        self.place_at_grid(label_bone_no_mound, 'E2', scale_factor=0.45) # Fix 36
        
        label_no_bone_mound = Text("No Bone & Mound", font_size=24)
        self.place_at_grid(label_no_bone_mound, 'B4', scale_factor=0.4) # Fix 38
        
        label_no_bone_no_mound = Text("No Bone & No Mound", font_size=24)
        self.place_at_grid(label_no_bone_no_mound, 'E4', scale_factor=0.45) # Fix 36

        self.add(full_square_content, square_outline, label_bone_mound, label_bone_no_mound, label_no_bone_mound, label_no_bone_no_mound)

        # === Animation for Lecture Line 1 ===
        # Gray out regions that don't match the evidence ('No Mound')
        self.play(self.lecture[0].animate.set_color(color_gray))
        self.play(
            rect_b.animate.set_fill(color_gray),
            rect_d.animate.set_fill(color_gray),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade out the impossible regions
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(color_a))
        self.play(
            FadeOut(rect_b), FadeOut(rect_d),
            FadeOut(label_bone_no_mound), FadeOut(label_no_bone_no_mound),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shift the remaining 'Mound' regions together
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        mound_group = VGroup(rect_a, rect_c).arrange(RIGHT, buff=0)
        self.play(
            mound_group.animate.move_to(square_outline.get_center()),
            label_bone_mound.animate.move_to(rect_a.get_center()).scale(0.8),
            label_no_bone_mound.animate.move_to(rect_c.get_center()).scale(0.8),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Re-scale remnants back into a full square
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GREEN))
        
        area_a = p_bone * p_m_b
        area_c = (1 - p_bone) * p_m_nb
        total = area_a + area_c
        
        new_w_a = (area_a / total) * sq_size
        new_w_c = (area_c / total) * sq_size
        
        target_rect_a = Rectangle(width=new_w_a, height=sq_size, fill_opacity=0.8, fill_color=color_a, stroke_width=1)
        target_rect_c = Rectangle(width=new_w_c, height=sq_size, fill_opacity=0.8, fill_color=color_c, stroke_width=1)
        target_group = VGroup(target_rect_a, target_rect_c).arrange(RIGHT, buff=0).move_to(square_outline.get_center())
        
        self.play(
            Transform(rect_a, target_rect_a),
            Transform(rect_c, target_rect_c),
            label_bone_mound.animate.move_to(target_rect_a.get_center()).scale(1.2),
            label_no_bone_mound.animate.move_to(target_rect_c.get_center()).scale(1.0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final label for the posterior probability
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(color_a))
        
        brace_bone = Brace(rect_a, DOWN, buff=0.1)
        text_post = Text("Posterior Probability", font_size=20)
        posterior_group = VGroup(brace_bone, text_post).arrange(DOWN, buff=0.1)
        # Issue 37 Fix:
        self.place_in_area(posterior_group, 'F2', 'F5', scale_factor=0.7)
        
        self.play(FadeIn(posterior_group))
        self.wait(2)
