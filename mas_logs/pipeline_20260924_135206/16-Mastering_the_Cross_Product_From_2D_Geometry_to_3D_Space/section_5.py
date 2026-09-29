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
        self.setup_layout("Summary and Quick Check", [
            "Cross product result is always a vector.",
            "Dot product result is always a scalar value.",
            "Changing vector order reverses the resulting vector direction."
        ])
        
        # Elements to display
        vec_icon = Arrow(start=ORIGIN, end=UP*1.5, color="#FF0000")
        scalar_icon = Dot(color="#0000FF", radius=0.2)
        reverse_icon = VGroup(
            Arrow(start=LEFT, end=RIGHT, color="#FFFF00"),
            Arrow(start=RIGHT, end=LEFT, color="#FFFF00")
        ).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.place_at_grid(vec_icon, 'B5', scale_factor=0.6)
        self.play(FadeIn(vec_icon), Flash(vec_icon, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#0000FF")
        self.place_at_grid(scalar_icon, 'C5', scale_factor=0.6)
        self.play(FadeIn(scalar_icon), Flash(scalar_icon, color="#0000FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(reverse_icon, 'D5', scale_factor=0.6)
        self.play(FadeIn(reverse_icon), Flash(reverse_icon, color="#FFFF00"))
        self.wait(1)

        # Final checkmark
        checkmark = Tex(r"$\checkmark$", color=GREEN).scale(2.0)
        self.place_at_grid(checkmark, 'E3', scale_factor=0.5)
        self.play(Create(checkmark))
        self.wait(2)
