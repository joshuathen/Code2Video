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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "The Final Verdict & Summary", 
            [
                "Oranges are seed-carrying fruits born on trees.", 
                "Tubers are underground stems packed with starch.", 
                "Clearly, an orange is definitely not a tuber!"
            ]
        )

        # === Animation for Lecture Line 1 ===
        # A summary list item for Orange appears in #FFFFFF.
        orange_summary = Text("Orange: Fruit", color="#FFFFFF", font_size=32)
        # Fix for Issue 40: Utilizing right panel area
        self.place_in_area(orange_summary, 'B3', 'B4', scale_factor=1.0)
        
        self.play(
            FadeIn(orange_summary),
            self.lecture[0].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A summary list item for Potato appears in #FFFFFF.
        potato_summary = Text("Potato: Tuber", color="#FFFFFF", font_size=32)
        # Fix for Issue 41: Improving vertical flow/alignment
        self.place_in_area(potato_summary, 'C3', 'C4', scale_factor=1.0)
        
        self.play(
            FadeIn(potato_summary),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Oliver the Owl reappears at the bottom, nodding. [Asset: Oliver the Owl]
        # The final verdict 'Orange != Tuber' (#FF0000) flashes in the center.
        
        # Representing Asset: Oliver the Owl
        oliver_owl = VGroup(
            Circle(color="#FFFFFF", fill_opacity=0.2, radius=0.4),
            Text("Oliver", color="#FFFFFF", font_size=18)
        ).arrange(ORIGIN) # Simple relative grouping
        self.place_in_area(oliver_owl, "E3", "F4")
        
        verdict_text = Text("Orange != Tuber", color="#FF0000", font_size=48)
        # Fix for Issue 42: Emphasis through horizontal span and scale
        self.place_in_area(verdict_text, 'D2', 'D5', scale_factor=1.1)

        # Oliver reappears
        self.play(FadeIn(oliver_owl))
        
        # Nodding animation (simulated by brief vertical movement)
        self.play(
            oliver_owl.animate.shift(UP * 0.15),
            run_time=0.25,
            rate_func=there_and_back
        )
        self.play(
            oliver_owl.animate.shift(UP * 0.15),
            run_time=0.25,
            rate_func=there_and_back
        )

        # Final verdict flash and lecture line color change to match
        self.play(
            self.lecture[2].animate.set_color("#FF0000"),
            Write(verdict_text)
        )
        self.play(Flash(verdict_text, color="#FF0000", flash_radius=1.5, num_lines=12))
        
        self.wait(3)
