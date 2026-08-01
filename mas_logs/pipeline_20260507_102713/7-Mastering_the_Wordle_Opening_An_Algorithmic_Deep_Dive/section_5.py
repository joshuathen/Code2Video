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
        self.setup_layout(
            "The Clash of Titans: ADIEU vs. CRANE vs. SALET", 
            [
                "Many players prioritize finding vowels with ADIEU.", 
                "However, consonants prune the word list more effectively.", 
                "SALET and CRANE outperform ADIEU in average steps."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        adieu = Text("ADIEU", font_size=32, color=WHITE)
        crane = Text("CRANE", font_size=32, color=WHITE)
        salet = Text("SALET", font_size=32, color=WHITE)
        
        # Word list appears
        self.place_at_grid(adieu, "B2")
        self.place_at_grid(crane, "C2")
        self.place_at_grid(salet, "D2")
        
        self.play(
            Write(adieu),
            Write(crane),
            Write(salet),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        
        # Horizontal bars for remaining possibilities (Issue 41: Column 4)
        bar_adieu = Rectangle(width=2.5, height=0.3, color=BLUE, fill_opacity=0.8)
        bar_crane = Rectangle(width=1.0, height=0.3, color=BLUE, fill_opacity=0.4)
        bar_salet = Rectangle(width=0.4, height=0.3, color=BLUE, fill_opacity=0.4)
        
        self.place_at_grid(bar_adieu, "B4")
        self.place_at_grid(bar_crane, "C4")
        self.place_at_grid(bar_salet, "D4")
        
        # Numerical labels (Issue 42: Column 5)
        label_50 = Text("50+ words", font_size=18, color=BLUE)
        label_20 = Text("< 20 words", font_size=18, color=BLUE)
        
        self.place_at_grid(label_50, "B5")
        self.place_at_grid(label_20, "D5")
        
        self.play(
            GrowFromEdge(bar_adieu, LEFT),
            GrowFromEdge(bar_crane, LEFT),
            GrowFromEdge(bar_salet, LEFT),
            FadeIn(label_50),
            FadeIn(label_20),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#D4AF37") # Gold
        
        # Asset: Leaderboard Icon (Issue 28)
        leaderboard_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/leaderboard.svg")
        self.place_at_grid(leaderboard_icon, "A4", scale_factor=0.6)

        # Leaderboard headers (Issue 40: header_word at B4)
        header_rank = Text("RANK", font_size=20, color="#D4AF37")
        header_word = Text("STRATEGY", font_size=20, color="#D4AF37")
        header_steps = Text("AVG STEPS", font_size=20, color="#D4AF37")
        
        self.place_at_grid(header_rank, "B2")
        self.place_at_grid(header_word, "B4", scale_factor=0.8)
        self.place_at_grid(header_steps, "B5")
        
        # Ranking text
        rank1 = Text("1st", font_size=24, color="#D4AF37")
        rank2 = Text("2nd", font_size=24, color="#D4AF37")
        rank3 = Text("3rd", font_size=24, color="#D4AF37")
        
        self.place_at_grid(rank1, "C2")
        self.place_at_grid(rank2, "D2")
        self.place_at_grid(rank3, "E2")
        
        # Average Steps text
        step1 = Text("3.42", font_size=24, color="#D4AF37")
        step2 = Text("3.48", font_size=24, color="#D4AF37")
        step3 = Text("3.66", font_size=24, color="#D4AF37")
        
        self.place_at_grid(step1, "C5")
        self.place_at_grid(step2, "D5")
        self.place_at_grid(step3, "E5")

        # Reorder transition into leaderboard: SALET (1st), CRANE (2nd), ADIEU (3rd)
        self.play(
            FadeOut(bar_adieu, bar_crane, bar_salet, label_50, label_20),
            adieu.animate.move_to(self.grid["E4"]).set_color("#D4AF37"),
            crane.animate.move_to(self.grid["D4"]).set_color("#D4AF37"),
            salet.animate.move_to(self.grid["C4"]).set_color("#D4AF37"),
            FadeIn(leaderboard_icon),
            FadeIn(header_rank, header_word, header_steps),
            FadeIn(rank1, rank2, rank3),
            FadeIn(step1, step2, step3),
            run_time=2
        )
        self.wait(3)
