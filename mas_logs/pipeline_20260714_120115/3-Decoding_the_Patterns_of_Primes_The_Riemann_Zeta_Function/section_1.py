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
        # --- Data Initialization ---
        title = "The Mystery of the Prime Hunters"
        lecture_lines = [
            "Prime numbers appear scattered randomly across the number line.",
            "Pete the explorer searches for order within this chaos.",
            "He needs a mathematical lens to reveal hidden patterns."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Paths for assets
        pete_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/explorer.svg"
        lens_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/lens.svg"

        # === Animation for Lecture Line 1 ===
        # Display a scattered sequence of prime numbers (2, 3, 5, 7...) in #FFFFFF.
        self.lecture[0].set_color(YELLOW)
        
        primes_text = ["2", "3", "5", "7", "11", "13", "17", "19"]
        prime_mobjects = VGroup(*[Text(p, color="#FFFFFF", font_size=36) for p in primes_text])
        
        # Grid positions: B2, C4, B5, D2, E5, C6, D4, B3
        grid_positions = ["B2", "C4", "B5", "D2", "E5", "C6", "D4", "B3"]
        for i, (obj, pos) in enumerate(zip(prime_mobjects, grid_positions)):
            # L006: Scale down near edges (C6)
            s_factor = 0.8 if pos != "C6" else 0.6
            self.place_at_grid(obj, pos, scale_factor=s_factor)
            
        self.play(FadeIn(prime_mobjects))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A character named 'Pete' [Asset: ...explorer.svg] appears in #00FF00.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        pete = SVGMobject(pete_path)
        pete.set_color("#00FF00")
        self.place_at_grid(pete, "D3", scale_factor=0.8)
        
        pete_label = Text("Pete", font_size=18, color="#00FF00")
        # L003: Label within 1 grid unit
        pete_label.next_to(pete, DOWN, buff=0.1)
        
        self.play(FadeIn(pete), Write(pete_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the scattered primes using a lens [Asset: ...lens.svg] in #FFFF00 to show their randomness.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        lens = SVGMobject(lens_path)
        lens.set_color("#FFFF00")
        self.place_at_grid(lens, "C3", scale_factor=0.8)
        
        # Highlights
        highlight_animations = [obj.animate.set_color("#FFFF00") for obj in prime_mobjects]
        
        self.play(
            FadeIn(lens),
            *highlight_animations
        )
        self.wait(2)
