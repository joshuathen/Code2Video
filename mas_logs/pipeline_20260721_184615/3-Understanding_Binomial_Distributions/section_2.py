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
        # Fetching title and lecture lines from storyboard
        title = "The BINS Criteria"
        lecture_lines = [
            "We use 'BINS' to check if a scenario fits.",
            "First, trials must have only two Binary outcomes.",
            "Second, every trial must be Independent of others.",
            "Third, there must be a fixed Number of trials.",
            "Finally, the probability 'p' must be the Same."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # [self.wait(2.0)] Fade in 'BINS Criteria' header in #FFFFFF.
        self.lecture[0].set_color("#FFFFFF") # Storyboard says header is #FFFFFF
        
        bins_header = Text("BINS Criteria", font_size=32, color="#FFFFFF")
        # Fix Issue 30: Place header in Row B to avoid crowding top title
        self.place_in_area(bins_header, "B2", "B5", scale_factor=0.8)
        
        self.play(FadeIn(bins_header))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # [self.wait(1.5)] Display 'Binary' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/binary.svg] with a checkmark in #00FF00.
        self.lecture[1].set_color("#00FF00")
        
        # Load asset (Issue 17)
        binary_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/binary.svg")
        binary_icon.set_height(0.6) # L025: avoid set_height with stretch
        binary_icon.set_color("#00FF00")
        
        binary_text = Text("Binary", font_size=28, color="#00FF00")
        # L022: Fallback to Text for checkmark symbol
        checkmark_b = Text("✓", font_size=30, color="#00FF00")
        
        row_b = VGroup(binary_icon, binary_text, checkmark_b).arrange(RIGHT, buff=0.4)
        # Fix Issue 31: Shift down to Row C
        self.place_in_area(row_b, "C2", "C5", scale_factor=0.8)
        
        self.play(FadeIn(binary_icon), Write(binary_text), FadeIn(checkmark_b))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # [self.wait(1.5)] Display 'Independent' with a checkmark in #00FF00.
        self.lecture[2].set_color("#00FF00")
        
        independent_text = Text("Independent", font_size=28, color="#00FF00")
        checkmark_i = Text("✓", font_size=30, color="#00FF00")
        
        row_i = VGroup(independent_text, checkmark_i).arrange(RIGHT, buff=0.4)
        # Fix Issue 31: Shift down to Row D
        self.place_in_area(row_i, "D2", "D5", scale_factor=0.8)
        
        self.play(Write(independent_text), FadeIn(checkmark_i))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # [self.wait(1.5)] Display 'Number (n=10)' with a checkmark in #00FF00.
        self.lecture[3].set_color("#00FF00")
        
        number_text = Text("Number (n=10)", font_size=28, color="#00FF00")
        checkmark_n = Text("✓", font_size=30, color="#00FF00")
        
        row_n = VGroup(number_text, checkmark_n).arrange(RIGHT, buff=0.4)
        # Fix Issue 31: Shift down to Row E
        self.place_in_area(row_n, "E2", "E5", scale_factor=0.8)
        
        self.play(Write(number_text), FadeIn(checkmark_n))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # [self.wait(1.5)] Display 'Same p' with a checkmark in #00FF00.
        self.lecture[4].set_color("#00FF00")
        
        same_p_text = Text("Same p", font_size=28, color="#00FF00")
        checkmark_s = Text("✓", font_size=30, color="#00FF00")
        
        row_s = VGroup(same_p_text, checkmark_s).arrange(RIGHT, buff=0.4)
        # Fix Issue 31: Shift down to Row F
        self.place_in_area(row_s, "F2", "F5", scale_factor=0.8)
        
        self.play(Write(same_p_text), FadeIn(checkmark_s))
        self.wait(1.5)
        
        self.wait(2.0)
