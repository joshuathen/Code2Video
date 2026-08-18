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
        lecture_lines = ["Notes are just musical fractions.", "A whole note is one.", "Four quarters make a whole."]
        self.setup_layout("The Math of Note Values (Fractions)", lecture_lines)
        
        # Assets
        # Note: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg is a placeholder
        # and doesn't actually exist as a useful asset. Using shapes instead.
        whole_circle = Circle(radius=1.5, color="#FF6347", fill_opacity=0.5)
        label_whole = Text("1", font_size=40).move_to(whole_circle.get_center())
        whole_note = VGroup(whole_circle, label_whole)

        half1 = Sector(radius=1.5, start_angle=PI/2, angle=PI, color="#1E90FF", fill_opacity=0.5)
        half2 = Sector(radius=1.5, start_angle=3*PI/2, angle=PI, color="#1E90FF", fill_opacity=0.5)
        
        q1 = Sector(radius=1.2, start_angle=0, angle=PI/2, color="#FFFF00", fill_opacity=0.5)
        q2 = Sector(radius=1.2, start_angle=PI/2, angle=PI/2, color="#FFFF00", fill_opacity=0.5)
        q3 = Sector(radius=1.2, start_angle=PI, angle=PI/2, color="#FFFF00", fill_opacity=0.5)
        q4 = Sector(radius=1.2, start_angle=3*PI/2, angle=PI/2, color="#FFFF00", fill_opacity=0.5)
        quarters = VGroup(q1, q2, q3, q4).arrange_in_grid(2, 2, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6347"))
        self.place_at_grid(whole_note, 'B3', scale_factor=0.5)
        self.play(FadeIn(whole_note))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#1E90FF"))
        self.play(FadeOut(whole_note))
        self.place_at_grid(half1, 'B2', scale_factor=0.5)
        self.place_at_grid(half2, 'B4', scale_factor=0.5)
        self.play(FadeIn(half1), FadeIn(half2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(FadeOut(half1), FadeOut(half2))
        self.place_in_area(quarters, 'C2', 'D4', scale_factor=0.6)
        self.play(FadeIn(quarters))
        self.wait(2)
