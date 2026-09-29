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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Practical Application: Why 'CRANE' or 'SALET'?", [
            "Starting words like CRANE are mathematically superior.",
            "They maximize vowel frequency and common consonants.",
            "This provides the highest information gain per turn."
        ])
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/crane.svg]
        crane_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/crane.svg")
        self.place_at_grid(crane_icon, "A6", scale_factor=0.3)
        self.add(crane_icon)

        # --- Preparation ---
        letters = ["A", "R", "N", "E", "S", "T"]
        frequencies = [0.12, 0.10, 0.09, 0.13, 0.11, 0.08]
        bars = VGroup()
        for i, (l, f) in enumerate(zip(letters, frequencies)):
            bar = Rectangle(height=f*10, width=0.5, color="#32CD32", fill_opacity=0.8)
            label = Text(l, font_size=20).next_to(bar, UP)
            bar_group = VGroup(bar, label)
            bars.add(bar_group)
        bars.arrange(RIGHT, buff=0.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Fix 29/44: Encroachment
        self.place_in_area(bars, 'D3', 'F6', scale_factor=0.6)
        self.play(Create(bars), run_time=2)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#32CD32"))
        word = Text("CRANE", font_size=48, color=WHITE)
        # Fix 30/45: Disjointed flow
        self.place_at_grid(word, 'B3', scale_factor=1.0)
        self.play(Write(word))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FF1493"))
        
        # Fix 31/46: Target overlap/anchoring
        target = Circle(radius=0.3, color="#FF1493", fill_opacity=0.5)
        self.place_at_grid(target, 'C3', scale_factor=0.8)
        target_label = Text("Max Entropy", font_size=20, color="#FF1493").next_to(target, RIGHT)
        
        self.play(
            word.animate.move_to(target.get_center()),
            FadeIn(target),
            Write(target_label)
        )
        self.wait(4)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/crane.svg]
        self.play(
            FadeOut(bars),
            FadeOut(word),
            FadeOut(target),
            FadeOut(target_label)
        )
        self.wait(2)
