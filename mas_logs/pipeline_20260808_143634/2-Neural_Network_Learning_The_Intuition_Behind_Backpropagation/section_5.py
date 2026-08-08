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
        lecture_lines = [
            "Forward passes predict while backward passes blame.",
            "Continuous weight updates drive the learning process.",
            "Repetition transforms basic inputs into high intelligence."
        ]
        self.setup_layout("Summary & Continuous Learning", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        checklist = VGroup(
            Text("1. Forward Pass", font_size=24, color=WHITE),
            Text("2. Calculate Loss", font_size=24, color=WHITE),
            Text("3. Backward Pass", font_size=24, color=WHITE),
            Text("4. Update Weights", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        self.place_in_area(checklist, 'A2', 'E3', scale_factor=0.8)
        self.play(FadeIn(checklist))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        # Highlight items one by one
        for i in range(4):
            self.play(checklist[i].animate.set_color("#00FF00"))
        
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(checklist))
        
        # Neural Network Asset
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        
        # Glowing Neural Network visual representation (moved to A4-E5 per feedback)
        dots = VGroup(*[Dot(radius=0.15, color=WHITE) for _ in range(10)])
        dots.arrange_in_grid(2, 5, buff=0.6)
        self.place_in_area(dots, 'A4', 'E5', scale_factor=0.7)
        
        # Add glow effect (moved to B3-D4 per feedback)
        glow = VGroup(*[Dot(radius=0.3, color=BLUE, fill_opacity=0.3) for _ in range(10)])
        glow.arrange_in_grid(2, 5, buff=0.6)
        self.place_in_area(glow, 'B3', 'D4', scale_factor=0.6)
        
        # Positioning Asset: Place brain near dots
        self.place_at_grid(brain, 'F5', scale_factor=0.5)
        
        self.play(FadeIn(glow), FadeIn(dots), FadeIn(brain))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
