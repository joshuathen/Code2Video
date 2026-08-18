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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Summary", [
            "Curves bridge different dimensional spaces.", 
            "They are infinitely long yet continuous.", 
            "Complexity emerges from simple recursive rules."
        ])
        
        # Visual assets
        # Summarize 1D to 2D
        dim1 = Line(LEFT, RIGHT, color=BLUE).scale(0.5)
        arrow = Arrow(LEFT, RIGHT, color=WHITE).scale(0.5)
        dim2 = Square(side_length=1.0, color=GREEN)
        dim_group = VGroup(dim1, arrow, dim2).arrange(RIGHT)
        # Fix 32, 34: Position to B5, scale 0.8
        self.place_at_grid(dim_group, 'B5', scale_factor=0.8)
        
        # Key takeaways list (mockup icons)
        takeaway_icons = VGroup(
            Dot(color=YELLOW),
            Dot(color=YELLOW),
            Dot(color=YELLOW)
        ).arrange(DOWN, buff=0.5)
        # Fix 33, 34: Position to E5, scale 0.8
        self.place_at_grid(takeaway_icons, 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(dim_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(FadeIn(takeaway_icons[0]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(FadeIn(takeaway_icons[1:]))
        self.wait(1)
        
        self.play(FadeOut(*self.mobjects))
