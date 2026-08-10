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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The 8 Axioms: Defining the Rules of the Playground", [
            "The playground requires two specific properties.",
            "First, closure: operations must stay inside.",
            "Second, internal structure: rules like associativity.",
            "Think of it like a vector animal shelter.",
            "If these hold, it is a vector space."
        ])
        
        # Create boxes for axioms
        boxes = VGroup(*[Square(side_length=0.8, color=WHITE) for _ in range(8)])
        for i, box in enumerate(boxes):
            row = chr(ord('B') + (i // 4))
            col = str((i % 4) + 2)
            self.place_at_grid(box, f"{row}{col}", scale_factor=0.6)

        # Load Assets
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg").scale(0.3)
        shelter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shelter.svg").scale(0.3)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(boxes))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ORANGE)
        
        # Using simple Text instead of LaTeX if \checkmark fails, or ensure proper spacing
        checkmarks1 = VGroup(*[Tex(r"$\checkmark$", color="#2ecc71").scale(1.0).move_to(boxes[i].get_center() + RIGHT*0.4) for i in range(4)])
        animal_icon.move_to(boxes[0].get_center() + LEFT*0.4)
        
        self.play(
            FadeIn(checkmarks1),
            FadeIn(animal_icon),
            *[boxes[i].animate.set_color("#e67e22") for i in range(4)]
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        checkmarks2 = VGroup(*[Tex(r"$\checkmark$", color="#2ecc71").scale(1.0).move_to(boxes[i].get_center() + RIGHT*0.4) for i in range(4, 8)])
        shelter_icon.move_to(boxes[4].get_center() + LEFT*0.4)
        
        self.play(
            FadeIn(checkmarks2),
            FadeIn(shelter_icon),
            *[boxes[i].animate.set_color("#3498db") for i in range(4, 8)]
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(PURPLE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        
        banner = Text("Vector Space Confirmed!", color=WHITE, font_size=32)
        self.place_in_area(banner, "F2", "F5", scale_factor=0.7)
        self.play(Write(banner), Indicate(boxes))
