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
        lecture_lines = ["CLT is the foundation of inference.", "We predict populations from small samples.", "Accurate estimates require minimal data points."]
        self.setup_layout("Why It Matters: Real-World Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a real-world scenario (smartphone battery example)
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        self.place_at_grid(phone, "C3", scale_factor=0.9)
        self.play(FadeIn(phone))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Show confidence intervals / Margin of Error
        interval = VGroup(
            Line(LEFT*1.5, RIGHT*1.5, color=WHITE),
            Dot(color=RED),
            Line(UP*0.2, DOWN*0.2, color=WHITE).shift(LEFT*1.0),
            Line(UP*0.2, DOWN*0.2, color=WHITE).shift(RIGHT*1.0)
        )
        self.place_in_area(interval, "D3", "E5", scale_factor=0.7)
        self.play(Create(interval))
        self.play(interval.animate.scale(0.6)) # Narrowing
        self.lecture[1].set_color("#87CEEB")

        # === Animation for Lecture Line 3 ===
        # Checkmark
        checkmark = Tex(r"$\checkmark$", color="#32CD32")
        self.place_at_grid(checkmark, "C4", scale_factor=0.8)
        
        # Population Inference text
        inf_text = Text("Population Inference", font_size=20, color=WHITE)
        inf_text.next_to(phone, DOWN, buff=0.2)
        
        self.play(GrowFromCenter(checkmark), Write(inf_text))
        self.lecture[2].set_color("#32CD32")
        self.wait(2)
