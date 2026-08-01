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
        # Title and Lecture lines setup
        title_text = "The Sign: Negative Determinants & Orientation"
        lecture_lines = [
            'Start with Pixel and our standard basis vectors.',
            'A negative determinant mirrors Pixel to the other side.',
            'Notice i-hat and j-hat have swapped their relative orientation.',
            'This swap means we have flipped the entire space.',
            'A negative sign indicates this fundamental change in orientation.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_I = "#FF0000"  # Red
        COLOR_J = "#00FF00"  # Green
        COLOR_PIXEL = "#FFFFFF" # White
        COLOR_HIGHLIGHT = "#FF00FF" # Magenta
        PIXEL_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create Pixel using SVGMobject
        pixel = SVGMobject(PIXEL_ASSET, color=COLOR_PIXEL).scale(0.6)
        
        # Basis Vectors
        i_vec = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=COLOR_I)
        j_vec = Arrow(start=ORIGIN, end=UP, buff=0, color=COLOR_J)
        i_label = Text("i", font_size=20, color=COLOR_I).next_to(i_vec.get_end(), RIGHT, buff=0.1)
        j_label = Text("j", font_size=20, color=COLOR_J).next_to(j_vec.get_end(), UP, buff=0.1)
        
        world = VGroup(pixel, i_vec, j_vec, i_label, j_label)
        # Fix Issue 44: Use B3 to E6
        self.place_in_area(world, 'B3', 'E6', scale_factor=1.0)
        
        self.play(FadeIn(world))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Mirroring Matrix: [[-1, 0], [0, 1]] applied to pixel and vectors
        pixel_flipped = pixel.copy().stretch(-1, 0)
        i_vec_flipped = Arrow(start=ORIGIN, end=LEFT, buff=0, color=COLOR_I)
        # Note: j_vec and j_label remain the same for this flip
        i_label_flipped = Text("i", font_size=20, color=COLOR_I).next_to(i_vec_flipped.get_end(), LEFT, buff=0.1)
        
        world_flipped = VGroup(pixel_flipped, i_vec_flipped, j_vec.copy(), i_label_flipped, j_label.copy())
        # Fix Issue 45: Use B3 to E6
        self.place_in_area(world_flipped, 'B3', 'E6', scale_factor=1.0)

        self.play(ReplacementTransform(world, world_flipped), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight that i is now to the left of j (flipped orientation).
        highlight_box = SurroundingRectangle(VGroup(i_label_flipped, j_label.copy()), color=COLOR_HIGHLIGHT, buff=0.2)
        self.play(Create(highlight_box))
        self.play(FadeOut(highlight_box))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Circular arrow showing orientation direction reversing
        # Initial CCW orientation (visual aid)
        orient_arrow_ccw = Arc(radius=0.6, start_angle=0, angle=PI/2, color=COLOR_J)
        orient_arrow_ccw.add_tip(tip_length=0.1)
        orient_arrow_ccw.move_to(world_flipped.get_center() + RIGHT*0.2)
        
        # Reversed CW orientation
        orient_arrow_cw = Arc(radius=0.6, start_angle=PI, angle=-PI/2, color=COLOR_HIGHLIGHT)
        orient_arrow_cw.add_tip(tip_length=0.1)
        orient_arrow_cw.move_to(world_flipped.get_center() + LEFT*0.2)

        self.play(FadeIn(orient_arrow_ccw))
        self.play(ReplacementTransform(orient_arrow_ccw, orient_arrow_cw))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        conclusion = Text("Negative Determinant = Flip", font_size=24, color=COLOR_HIGHLIGHT)
        # Fix Issue 46: Use place_in_area for centering
        self.place_in_area(conclusion, 'F3', 'F5', scale_factor=1.1)
        
        self.play(Write(conclusion))
        self.wait(2)

        # Final cleanup: Reset colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
