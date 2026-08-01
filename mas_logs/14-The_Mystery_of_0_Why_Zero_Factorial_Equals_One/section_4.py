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
        # Setup the layout with specific lines
        lines = [
            "Factorials count how many ways to arrange objects.",
            "Three objects have six arrangements; one has one.",
            "An empty shelf has exactly one arrangement."
        ]
        self.setup_layout("The Combinatorial Perspective (Arrangements)", lines)

        # Helper for a simple hat shape
        def create_hat(color="#FF00FF"):
            brim = Line(LEFT * 0.4, RIGHT * 0.4, stroke_width=4, color=color)
            top = Arc(radius=0.25, start_angle=0, angle=PI, stroke_width=4, color=color).shift(UP * 0.05)
            return VGroup(brim, top)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Display 3 distinct magenta hats (#FF00FF)
        hats = VGroup(create_hat(), create_hat(), create_hat())
        # Position hats on a virtual shelf line
        self.place_at_grid(hats[0], "B2", scale_factor=0.8)
        self.place_at_grid(hats[1], "B3", scale_factor=0.8)
        self.place_at_grid(hats[2], "B4", scale_factor=0.8)

        # Display the number '6' (3!)
        # Use Text instead of MathTex for robustness in environment
        arrangements_text = Text("6 (3!)", color=WHITE, font_size=32)
        self.place_at_grid(arrangements_text, "C3", scale_factor=1.0)

        self.play(FadeIn(hats), Write(arrangements_text))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Remove all hats leaving an empty shelf
        # The shelf is a line at row D
        shelf = Line(self.grid["D1"], self.grid["D5"], stroke_width=4, color=GREY_B)
        
        # Display text '1 Way' (#00FFFF) next to the empty shelf
        one_way_text = Text("1 Way", color="#00FFFF", font_size=32)
        self.place_at_grid(one_way_text, "E3", scale_factor=1.0)

        # Animation transition: hats and 3! fade out, shelf and "1 Way" appear
        self.play(
            FadeOut(hats),
            FadeOut(arrangements_text),
            Create(shelf)
        )
        self.play(Write(one_way_text))
        self.wait(4)

        # Final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
