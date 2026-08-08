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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition: The Tip-to-Tail Rule", [
            "Add vectors by connecting them tip-to-tail.", 
            "The resultant vector connects start to end.", 
            "Addition forms the diagonal of a parallelogram."
        ])
        
        # Define grid-aware components
        # Original coordinates are relative to origin, we will shift them to our grid area.
        v_start = ORIGIN
        v_end = np.array([1.5, 0.5, 0])
        w_end = np.array([0.5, 1.5, 0])
        
        # Group setup
        v = Arrow(v_start, v_end, color=WHITE, buff=0)
        w = Arrow(v_start, w_end, color=WHITE, buff=0)
        v_label = MathTex("v", color=WHITE)
        w_label = MathTex("w", color=WHITE)
        
        # Initial container group
        vector_group = VGroup(v, w, v_label, w_label)
        self.place_in_area(vector_group, 'A4', 'C6', scale_factor=0.6)
        
        # Ensure labels are near vectors after placement
        v_label.next_to(v.get_center(), UP, buff=0.1)
        w_label.next_to(w.get_center(), RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(Create(v), Create(w), Write(v_label), Write(w_label))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        w_shifted = Arrow(v_end, v_end + w_end, color=WHITE, buff=0)
        w_label_shifted = MathTex("w", color=WHITE)
        
        # Add labels to the same grid area as vectors
        # Note: Need to manage relative positions inside the group scale
        
        self.play(ReplacementTransform(w, w_shifted), ReplacementTransform(w_label, w_label_shifted))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        resultant = Arrow(v_start, v_end + w_end, color="#00FFFF", buff=0)
        res_label = MathTex("v+w", color="#00FFFF")
        self.place_at_grid(res_label, 'B5', scale_factor=0.7)
        
        dashed_v = DashedLine(w_end, v_end + w_end, color=GRAY)
        dashed_w = DashedLine(v_end, v_end + w_end, color=GRAY)
        
        self.play(Create(resultant), Write(res_label))
        self.play(Create(dashed_v), Create(dashed_w))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
