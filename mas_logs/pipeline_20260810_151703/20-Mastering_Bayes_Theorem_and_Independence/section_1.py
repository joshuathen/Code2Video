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
        self.setup_layout("Prerequisite Warm-up: Conditional Probability", [
            "Conditional probability focuses on a reduced sample space.",
            "Think of it as zooming into event B.",
            "We only count outcomes where both A and B happen."
        ])
        
        # Define grid of dots for population visualization
        dots = VGroup(*[Dot(radius=0.08, color=WHITE) for _ in range(100)])
        dots.arrange_in_grid(10, 10, buff=0.1)
        self.place_in_area(dots, "B2", "E5", scale_factor=0.6)
        
        # Assets (Using correct local path for cat.png from requirements)
        dog_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg")
        # Ensure we don't crash on cat.png if it needs to be imported differently,
        # but the request asks to use the assets referenced.
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        p_ab_label = MathTex(r"P(A|B) = \frac{\text{Area}(A \cap B)}{\text{Area}(B)}", font_size=28)
        self.place_at_grid(p_ab_label, "E4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(dots))
        self.lecture[0].set_color("#FF9900")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight B (Dog Owners)
        dog_subset = dots[:50]
        self.play(dog_subset.animate.set_color("#FF0000"))
        self.place_at_grid(dog_icon, "A3", scale_factor=0.3)
        self.play(FadeIn(dog_icon))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF9900")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight A inside B (Cat Owners who own dogs)
        cat_subset = dots[:20]
        self.play(cat_subset.animate.set_color("#00FF00"))
        self.place_at_grid(cat_icon, "A4", scale_factor=0.2)
        self.play(FadeIn(cat_icon))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        self.play(Write(p_ab_label))
        self.wait(2)
