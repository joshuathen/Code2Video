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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Trinity: Query, Key, and Value (Q, K, V)"
        lines = [
            "Attention uses three distinct roles: Query, Key, and Value.",
            "Query represents the specific information we seek.",
            "Key acts as a label to match against Query.",
            "Similarity scores determine which Key matches the Query best.",
            "Value provides the actual information for the winning match."
        ]
        self.setup_layout(title, lines)

        # Colors
        QUERY_COLOR = "#FFD700"
        KEY_COLOR = "#ADFF2F"
        VALUE_COLOR = "#00BFFF"
        GRAY_COLOR = GRAY
        HIGHLIGHT_COLOR = WHITE

        # Assets
        DOG_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg"
        BALL_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"

        # === Animation for Lecture Line 1 ===
        # "Attention uses three distinct roles: Query, Key, and Value."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        q_label = Text("Query", color=QUERY_COLOR, font_size=24)
        k_label = Text("Key", color=KEY_COLOR, font_size=24)
        v_label = Text("Value", color=VALUE_COLOR, font_size=24)
        
        self.place_at_grid(q_label, "A2")
        self.place_at_grid(k_label, "A4")
        self.place_at_grid(v_label, "A6")
        
        self.play(FadeIn(q_label), FadeIn(k_label), FadeIn(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Query represents the specific information we seek."
        self.play(
            self.lecture[0].animate.set_color(GRAY_COLOR),
            self.lecture[1].animate.set_color(QUERY_COLOR)
        )
        
        # Load and place Dog Icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg]
        dog_icon = SVGMobject(DOG_ASSET, color=QUERY_COLOR).scale(0.5)
        self.place_at_grid(dog_icon, "B2")
        
        query_sub_label = Text("Seeking 'Food'", font_size=16, color=QUERY_COLOR)
        query_sub_label.next_to(dog_icon, DOWN, buff=0.2)
        
        self.play(FadeIn(dog_icon), Write(query_sub_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Key acts as a label to match against Query."
        self.play(
            self.lecture[1].animate.set_color(GRAY_COLOR),
            self.lecture[2].animate.set_color(KEY_COLOR)
        )
        
        # Key 1: Ball [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg]
        key_ball_icon = SVGMobject(BALL_ASSET, color=KEY_COLOR).scale(0.4)
        key_ball_text = Text("Ball", font_size=16, color=KEY_COLOR)
        key_ball = VGroup(key_ball_icon, key_ball_text).arrange(RIGHT, buff=0.1)
        self.place_at_grid(key_ball, "C4") # Fix per Issue 26
        
        # Key 2: Leash
        key_leash_text = Text("Leash", font_size=16, color=KEY_COLOR)
        key_leash_box = Rectangle(height=0.3, width=0.5, color=KEY_COLOR, fill_opacity=0.2)
        key_leash = VGroup(key_leash_box, key_leash_text).arrange(RIGHT, buff=0.1)
        self.place_at_grid(key_leash, "D4") # Fix per Issue 28
        
        # Key 3: Kibble
        key_kibble_text = Text("Kibble", font_size=16, color=KEY_COLOR)
        key_kibble_box = Rectangle(height=0.3, width=0.5, color=KEY_COLOR, fill_opacity=0.2)
        key_kibble = VGroup(key_kibble_box, key_kibble_text).arrange(RIGHT, buff=0.1)
        self.place_at_grid(key_kibble, "E4") # Fix per Issue 27
        
        self.play(
            LaggedStart(
                FadeIn(key_ball),
                FadeIn(key_leash),
                FadeIn(key_kibble),
                lag_ratio=0.5
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Similarity scores determine which Key matches the Query best."
        self.play(
            self.lecture[2].animate.set_color(GRAY_COLOR),
            self.lecture[3].animate.set_color(QUERY_COLOR)
        )
        
        # Beam from Query (Dog) to Kibble Key
        beam = Line(dog_icon.get_right(), key_kibble.get_left(), color=QUERY_COLOR, stroke_width=4)
        match_label = Text("Match Found!", font_size=18, color=QUERY_COLOR)
        match_label.next_to(key_kibble, UP, buff=0.1)
        
        self.play(Create(beam))
        self.play(
            Write(match_label),
            key_kibble.animate.set_color(WHITE).scale(1.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Value provides the actual information for the winning match."
        self.play(
            self.lecture[3].animate.set_color(GRAY_COLOR),
            self.lecture[4].animate.set_color(VALUE_COLOR)
        )
        
        # Value: Book icon representing 'Food'
        book_rect = Rectangle(width=0.6, height=0.8, color=VALUE_COLOR, fill_opacity=0.3)
        book_line1 = Line(LEFT, RIGHT, color=VALUE_COLOR).scale(0.2).move_to(book_rect.get_center() + UP*0.1)
        book_line2 = Line(LEFT, RIGHT, color=VALUE_COLOR).scale(0.2).move_to(book_rect.get_center())
        book_line3 = Line(LEFT, RIGHT, color=VALUE_COLOR).scale(0.2).move_to(book_rect.get_center() + DOWN*0.1)
        book_icon = VGroup(book_rect, book_line1, book_line2, book_line3)
        
        value_text = Text("Food", font_size=18, color=VALUE_COLOR)
        value_group = VGroup(book_icon, value_text).arrange(DOWN, buff=0.1)
        
        # Place Value next to the Kibble match (same row E)
        self.place_at_grid(value_group, "E6")
        
        # Value Glow
        glow = Circle(radius=0.6, color=VALUE_COLOR, fill_opacity=0.4).set_stroke(width=0)
        glow.move_to(value_group.get_center())
        
        self.play(FadeIn(value_group))
        self.play(
            FadeIn(glow),
            value_group.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)

# Update Issues:
# update_issue(21, under_review=True, resolution_note="Integrated Dog and Ball SVG assets and aligned with storyboard.")
# update_issue(26, under_review=True, resolution_note="Moved key_ball to C4 as requested.")
# update_issue(27, under_review=True, resolution_note="Moved key_kibble to E4 to align with Value 'Food' on row E.")
# update_issue(28, under_review=True, resolution_note="Moved key_leash to D4 for vertical alignment under Key header.")
