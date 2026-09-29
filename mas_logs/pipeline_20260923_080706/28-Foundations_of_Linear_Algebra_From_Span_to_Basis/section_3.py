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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Dependence means one vector is redundant.", "Redundant vectors lie on the existing span.", "Dependent vectors add no new reachable space."]
        self.setup_layout("Linear Dependence: The Redundancy Rule", lecture_lines)
        
        # Define vectors
        v1 = Arrow(ORIGIN, [1, 1, 0], color=BLUE)
        v2 = Arrow(ORIGIN, [1.5, -0.5, 0], color=GREEN)
        v3 = Arrow(ORIGIN, [2.5, 0.5, 0], color=YELLOW) # v1 + v2
        
        vectors = VGroup(v1, v2, v3)
        # Apply fix from video critic: use C4 and scale factor 0.5 for best spacing/grid alignment
        self.place_at_grid(vectors, 'C4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        # Display three vectors where one is a sum. (Color: #FF33FF)
        self.play(Create(v1), Create(v2), Create(v3), run_time=1)
        self.play(self.lecture[0].animate.set_color("#FF33FF"))

        # === Animation for Lecture Line 2 ===
        # Flash the redundant vector to highlight dependency. (Color: #FF0000)
        self.play(Flash(v3.get_end(), color=RED, line_length=0.2), run_time=1)
        self.play(self.lecture[1].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 3 ===
        # Fade out the redundant vector to show subspace. (Color: #808080)
        self.play(FadeOut(v3), run_time=1)
        self.play(self.lecture[2].animate.set_color("#808080"))
        self.wait(1)
