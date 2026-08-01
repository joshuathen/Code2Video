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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title_str = "Prerequisite: The Vector as a Path"
        lecture_lines = [
            'Imagine a vector as a path from the origin.',
            "Meet i-hat and j-hat, our grid's unit steps.",
            'Every vector is a combination of these steps.'
        ]
        self.setup_layout(title_str, lecture_lines)

        # Color constants
        GRID_COLOR = "#222222"
        IHAT_COLOR = "#58C4DD"
        JHAT_COLOR = "#FC6255"
        VEC_COLOR = "#83C167"
        PIXEL_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg"

        # === Animation for Lecture Line 1 ===
        # Create a dark grid #222222
        grid_lines = VGroup()
        for row in ["A", "B", "C", "D", "E", "F"]:
            grid_lines.add(Line(self.grid[f"{row}1"], self.grid[f"{row}6"], color=GRID_COLOR, stroke_width=1))
        for col in ["1", "2", "3", "4", "5", "6"]:
            grid_lines.add(Line(self.grid[f"A{col}"], self.grid[f"F{col}"], color=GRID_COLOR, stroke_width=1))
        
        # Origin is E2. Draw axes to emphasize.
        x_axis = Arrow(self.grid['E1'], self.grid['E6'], color=GREY_E, buff=0, stroke_width=2, tip_length=0.15)
        y_axis = Arrow(self.grid['F2'], self.grid['A2'], color=GREY_E, buff=0, stroke_width=2, tip_length=0.15)
        
        # Pixel icon at origin (E2)
        pixel = SVGMobject(PIXEL_ASSET)
        self.place_at_grid(pixel, 'E2', scale_factor=0.4)
        pixel_label = Text("Pixel", color=WHITE, font_size=20)
        self.place_at_grid(pixel_label, 'F1', scale_factor=0.7) # Issue 46: Fix origin overlap

        self.lecture[0].set_color(WHITE)
        self.play(Create(grid_lines), Create(x_axis), Create(y_axis), run_time=1)
        self.play(FadeIn(pixel), Write(pixel_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # i-hat (E2 to E3) and j-hat (E2 to D2)
        i_hat = Arrow(self.grid['E2'], self.grid['E3'], color=IHAT_COLOR, buff=0, stroke_width=5)
        j_hat = Arrow(self.grid['E2'], self.grid['D2'], color=JHAT_COLOR, buff=0, stroke_width=5)
        
        i_hat_label = Text("i-hat", color=IHAT_COLOR, font_size=18)
        self.place_at_grid(i_hat_label, 'F3', scale_factor=0.6) # Issue 47: Fix spacing
        
        j_hat_label = Text("j-hat", color=JHAT_COLOR, font_size=18)
        self.place_at_grid(j_hat_label, 'D1', scale_factor=0.6) # Issue 47: Fix spacing

        self.lecture[1].set_color(IHAT_COLOR) # Using blue for line 2
        self.play(GrowArrow(i_hat), Write(i_hat_label), run_time=1)
        self.play(GrowArrow(j_hat), Write(j_hat_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Path to (3,2) which is C5. Move Pixel along path.
        path_x = Line(self.grid['E2'], self.grid['E5'], color=WHITE, stroke_width=2, stroke_opacity=0.5)
        path_y = Line(self.grid['E5'], self.grid['C5'], color=WHITE, stroke_width=2, stroke_opacity=0.5)
        
        main_vec = Arrow(self.grid['E2'], self.grid['C5'], color=VEC_COLOR, buff=0, stroke_width=6)
        vector_coords = Text("[3, 2]", color=VEC_COLOR, font_size=22)
        self.place_at_grid(vector_coords, 'C6', scale_factor=0.7) # Issue 45: Fix terminal point overlap

        self.lecture[2].set_color(VEC_COLOR)
        
        # Animate Pixel moving along the steps
        self.play(
            pixel.animate.move_to(self.grid['E5']),
            Create(path_x),
            run_time=1
        )
        self.play(
            pixel.animate.move_to(self.grid['C5']),
            Create(path_y),
            pixel_label.animate.move_to(self.grid['B5']).scale(0.6/0.7), # Issue 45: Reposition pixel label
            run_time=1
        )
        
        # Grow the final vector
        self.play(
            GrowArrow(main_vec),
            Write(vector_coords),
            run_time=1.5
        )
        self.play(FadeOut(path_x), FadeOut(path_y), run_time=1)
        self.wait(3)
